"""
================================================================================
چارچوب جامع مدیریت تطبیقی منابع در محیط‌های رایانش لبه بدون سرور (HybridEdge / FGRL)
تلفیق پیش‌بینی سری‌های زمانی چندگامی، برآورد عدم‌قطعیت و یادگیری تقویتی عمیق (PPO)
مطابق با فصل ۳ و ۴ رساله
================================================================================
"""

import os
import math
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# --- ۱. تشخیص انحراف مفهومی و پیش‌پردازش ---

class PageHinkley:
    """آزمون آماری Page-Hinkley جهت تشخیص Concept Drift"""
    def __init__(self, delta: float = 0.005, lambd: float = 0.05, alpha: float = 0.999):
        self.delta = delta
        self.lambd = lambd
        self.alpha = alpha
        self.mean = 0.0
        self.cum = 0.0
        self.cummin = 0.0
        self.n = 0

    def reset(self):
        self.mean = 0.0
        self.cum = 0.0
        self.cummin = 0.0
        self.n = 0

    def update(self, x: float) -> bool:
        self.n += 1
        if self.n == 1:
            self.mean = float(x)
            return False
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * x
        self.cum += (x - self.mean - self.delta)
        self.cummin = min(self.cummin, self.cum)
        drift_detected = (self.cum - self.cummin) > self.lambd
        if drift_detected:
            self.cum = 0.0
            self.cummin = 0.0
        return drift_detected


class MinMaxScaler1D:
    """نرمال‌سازی داده‌ها در بازه [0, 1]"""
    def __init__(self):
        self.lo = 0.0
        self.hi = 1.0

    def fit(self, x: np.ndarray):
        self.lo = float(np.min(x))
        self.hi = float(np.max(x))
        if self.hi - self.lo < 1e-12:
            self.hi = self.lo + 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.lo) / (self.hi - self.lo)).astype(np.float32)

    def inverse(self, x_scaled: np.ndarray) -> np.ndarray:
        return (x_scaled * (self.hi - self.lo) + self.lo).astype(np.float32)


def generate_synthetic_azure_trace(n_points: int = 3500) -> np.ndarray:
    """تولید داده‌های جایگزین بار کاری در صورت عدم وجود داده خام واقعی"""
    np.random.seed(42)
    t = np.linspace(0, 50, n_points)
    base = 100 + 40 * np.sin(t) + 20 * np.cos(2.5 * t)
    bursts = np.random.choice([0, 1], size=n_points, p=[0.95, 0.05]) * np.random.uniform(50, 150, n_points)
    noise = np.random.normal(0, 10, n_points)
    series = np.maximum(0.0, base + bursts + noise)
    return series.astype(np.float32)


def preprocess_workload_series(series: np.ndarray) -> np.ndarray:
    """پایپ‌لاین کامل پاک‌سازی و هموارسازی داده‌ها به روش IQR"""
    series = np.maximum(0.0, series)
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    upper_bound = q3 + 3.0 * iqr
    series = np.clip(series, 0, upper_bound)
    series_smoothed = pd.Series(series).rolling(window=5, min_periods=1).mean().to_numpy()
    return series_smoothed.astype(np.float32)


def train_test_split_series(series: np.ndarray, train_ratio: float = 0.7):
    """تقسیم زمانی داده‌ها به نسبت ۷۰٪ آموزش و ۳۰٪ آزمون"""
    n_train = int(len(series) * train_ratio)
    return series[:n_train], series[n_train:]


# --- ۲. مدل‌های پیش‌بینی سری زمانی و Monte-Carlo Dropout ---

def make_supervised_multi(data: np.ndarray, look_back: int, out_steps: int):
    X, y = [], []
    limit = len(data) - look_back - out_steps + 1
    for i in range(limit):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back : i + look_back + out_steps])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y


class LSTMForecaster(nn.Module):
    def __init__(self, look_back: int, out_steps: int = 3, hidden: int = 64, dropout_p: float = 0.2):
        super().__init__()
        self.look_back = look_back
        self.out_steps = out_steps
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2, batch_first=True, dropout=dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden, out_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])
        return self.fc(h)


class BiLSTMForecaster(nn.Module):
    def __init__(self, look_back: int, out_steps: int = 3, hidden: int = 64, dropout_p: float = 0.2):
        super().__init__()
        self.look_back = look_back
        self.out_steps = out_steps
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(2 * hidden, out_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])
        return self.fc(h)


class GRUForecaster(nn.Module):
    def __init__(self, look_back: int, out_steps: int = 3, hidden: int = 64, dropout_p: float = 0.2):
        super().__init__()
        self.look_back = look_back
        self.out_steps = out_steps
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=2, batch_first=True, dropout=dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden, out_steps)

    def forward(self, x):
        out, _ = self.gru(x)
        h = self.dropout(out[:, -1, :])
        return self.fc(h)


def train_deep_forecaster(model: nn.Module, data: np.ndarray, look_back: int, out_steps: int, epochs: int = 10, lr: float = 1e-3):
    X, y = make_supervised_multi(data, look_back, out_steps)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    bs = 64

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(X.size(0))
        for i in range(0, X.size(0), bs):
            idx = perm[i:i + bs]
            xb, yb = X[idx], y[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def rolling_forecasts_multi_mc(model: nn.Module, data: np.ndarray, look_back: int, out_steps: int, mc_runs: int = 20):
    n = len(data)
    F_mean = np.zeros((n, out_steps), dtype=np.float32)
    F_std = np.zeros((n, out_steps), dtype=np.float32)

    model.train() # فعال نگه‌داشتن Dropout جهت استنتاج Monte-Carlo
    with torch.no_grad():
        for t in range(look_back, n):
            window = data[t - look_back : t]
            x_in = torch.tensor(window, dtype=torch.float32).view(1, look_back, 1)
            mc_preds = [model(x_in).numpy().flatten() for _ in range(mc_runs)]
            mc_preds = np.array(mc_preds)
            F_mean[t, :] = np.clip(np.mean(mc_preds, axis=0), 0.0, 1.0)
            F_std[t, :] = np.std(mc_preds, axis=0)

    for t in range(look_back):
        F_mean[t, :] = data[t]
        F_std[t, :] = 0.01

    model.eval()
    return F_mean, F_std


def moving_average_forecast(series: np.ndarray, look_back: int, out_steps: int, window: int = 12):
    n = len(series)
    Fm = np.zeros((n, out_steps), dtype=np.float32)
    Fs = np.zeros_like(Fm)
    for t in range(n):
        start = max(0, t - window + 1)
        Fm[t, :] = float(np.mean(series[start : t + 1]))
    return Fm, Fs


def arima_forecast(series: np.ndarray, look_back: int, out_steps: int, order=(2, 1, 1)):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        n = len(series)
        Fm = np.zeros((n, out_steps), dtype=np.float32)
        Fs = np.zeros_like(Fm)
        res = SARIMAX(series, order=order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        for t in range(look_back, n):
            start, end = t + 1, t + out_steps
            if start < n:
                pred = res.get_prediction(start=start, end=min(end, n - 1), dynamic=False)
                mu = np.asarray(pred.predicted_mean, dtype=np.float32)
                if mu.size < out_steps:
                    mu = np.pad(mu, (0, out_steps - mu.size), mode="edge")
                Fm[t, :] = np.clip(mu[:out_steps], 0.0, 1.0)
            else:
                Fm[t, :] = series[-1]
        return Fm, Fs
    except Exception:
        return moving_average_forecast(series, look_back, out_steps)


# --- ۳. محیط شبیه‌سازی Gymnasium (EdgeEnv) ---

class EdgeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: np.ndarray, forecast_mean: np.ndarray, forecast_std: np.ndarray = None,
                 look_back: int = 24, k: int = 3, latency_w: float = 3.0, waste_w: float = 1.0,
                 under_w: float = 2.0, unc_w: float = 0.5, eff_w: float = 0.3, sla_threshold: float = 0.5):
        super().__init__()
        self.data = data
        self.forecast_mean = forecast_mean
        self.forecast_std = forecast_std if forecast_std is not None else np.zeros_like(forecast_mean)
        self.look_back = look_back
        self.k = k
        self.n_steps = len(data)

        self.alpha = waste_w
        self.beta = under_w
        self.gamma = latency_w
        self.delta = unc_w
        self.epsilon = eff_w
        self.sla_threshold = sla_threshold

        self.action_space = spaces.Discrete(5) # {Scale Down, Keep, Scale Up, Offload, Migrate}
        obs_dim = 5 + 2 * k
        self.observation_space = spaces.Box(low=-10.0, high=100.0, shape=(obs_dim,), dtype=np.float32)

        self.current_step = self.look_back
        self.allocated_resource = 0.3
        self.reset()

    def _obs(self):
        t = self.current_step
        u_t = float(self.data[t])
        c_t = float(self.allocated_resource)
        w_future = self.forecast_mean[t, :self.k]
        sigma_future = self.forecast_std[t, :self.k]

        delta_w = float(self.data[t] - self.data[t - 1]) if t > 0 else 0.0
        stress = float(u_t / (c_t + 1e-5))
        volatility = float(np.std(self.data[max(0, t - 10) : t + 1])) if t >= 10 else 0.0

        return np.concatenate([
            np.array([u_t, c_t], dtype=np.float32),
            w_future.astype(np.float32),
            sigma_future.astype(np.float32),
            np.array([delta_w, stress, volatility], dtype=np.float32)
        ])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.look_back
        self.allocated_resource = float(np.clip(self.data[self.current_step] + 0.05, 0.05, 1.0))
        return self._obs(), {}

    def step(self, action: int):
        t = self.current_step
        u_t = float(self.data[t])

        if action == 0:   # Scale Down
            self.allocated_resource = max(0.05, self.allocated_resource - 0.1)
        elif action == 1: # Keep
            pass
        elif action == 2: # Scale Up
            self.allocated_resource = min(1.0, self.allocated_resource + 0.1)
        elif action == 3: # Offload
            self.allocated_resource = min(1.0, max(self.allocated_resource, u_t * 0.9))
        elif action == 4: # Migrate
            self.allocated_resource = min(1.0, max(self.allocated_resource, u_t))

        c_t = self.allocated_resource
        over = max(0.0, c_t - u_t)
        under = max(0.0, u_t - c_t)

        latency = (under / (c_t + 1e-5)) + (0.05 if action == 3 else 0.0)
        sla_violation = 1.0 if latency > self.sla_threshold else (latency / self.sla_threshold)
        unc_risk = float(np.mean(self.forecast_std[t, :self.k]))
        eff_bonus = max(0.0, 1.0 - (abs(c_t - u_t) / (u_t + 1e-5)))

        reward = -(
            self.alpha * (over ** 2) +
            self.beta * (under ** 2) +
            self.gamma * sla_violation +
            self.delta * unc_risk -
            self.epsilon * eff_bonus
        )

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - self.k - 1

        info = {
            "u_t": u_t, "c_t": c_t, "over": over, "under": under,
            "latency": latency, "sla_violation": sla_violation > 0.5,
            "reward": reward, "action": action
        }

        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, False, info


# --- ۴. استخراج ویژگی و کال‌بک‌های PPO ---

class FusionFeaturesExtractor(BaseFeaturesExtractor):
    """استخراج ویژگی چندوجهی (Multimodal Feature Fusion)"""
    def __init__(self, observation_space: spaces.Box, look_back: int = 24, k: int = 3, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)
        self.k = k
        self.state_fc = nn.Sequential(nn.Linear(5, 32), nn.ReLU())
        self.forecast_fc = nn.Sequential(nn.Linear(k, 32), nn.ReLU())
        self.unc_fc = nn.Sequential(nn.Linear(k, 16), nn.ReLU())
        self.fusion_head = nn.Sequential(nn.Linear(32 + 32 + 16, features_dim), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        u_c = obs[:, :2]
        w_future = obs[:, 2 : 2 + self.k]
        sigma_future = obs[:, 2 + self.k : 2 + 2 * self.k]
        extra_feats = obs[:, 2 + 2 * self.k :]

        f_state = self.state_fc(torch.cat([u_c, extra_feats], dim=1))
        f_forecast = self.forecast_fc(w_future)
        f_unc = self.unc_fc(sigma_future)
        return self.fusion_head(torch.cat([f_state, f_forecast, f_unc], dim=1))


class EntropyTargetCallback(BaseCallback):
    """تنظیم تطبیقی آنتروپی سیاست PPO"""
    def __init__(self, target_entropy: float = None, lr: float = 1e-3, verbose: int = 0):
        super().__init__(verbose)
        self.target_entropy = target_entropy
        self.lr = lr
        self.ent_coef = 0.01

    def _on_training_start(self) -> None:
        if self.target_entropy is None:
            self.target_entropy = 0.5 * math.log(self.training_env.action_space.n)

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            current_ent = getattr(self.model, "ent_coef", 0.01)
            if isinstance(current_ent, (float, int)):
                diff = self.target_entropy - current_ent
                self.ent_coef = float(np.clip(self.ent_coef + self.lr * diff, 1e-4, 0.1))
                self.model.ent_coef = self.ent_coef
        return True


class OnlineLSTMCallback(BaseCallback):
    """پایش Page-Hinkley و ریزآموزی برخط (Online Fine-Tuning)"""
    def __init__(self, env_ref: EdgeEnv, lstm_model: nn.Module, look_back: int = 24, k: int = 3,
                 update_every: int = 1500, epochs: int = 2, lr: float = 1e-4, verbose: int = 0):
        super().__init__(verbose)
        self.env_ref = env_ref
        self.lstm_model = lstm_model
        self.look_back = look_back
        self.k = k
        self.update_every = update_every
        self.epochs = epochs
        self.lr = lr
        self.ph_detector = PageHinkley()

    def _on_step(self) -> bool:
        t = self.env_ref.current_step
        if t < len(self.env_ref.data):
            val = self.env_ref.data[t]
            drift = self.ph_detector.update(val)
            if drift or (self.num_timesteps > 0 and self.num_timesteps % self.update_every == 0):
                recent_data = self.env_ref.data[max(0, t - 200) : t]
                if len(recent_data) > self.look_back + self.k + 10:
                    X, y = make_supervised_multi(recent_data, self.look_back, self.k)
                    opt = optim.Adam(self.lstm_model.parameters(), lr=self.lr)
                    loss_fn = nn.MSELoss()
                    self.lstm_model.train()
                    for _ in range(self.epochs):
                        opt.zero_grad()
                        loss = loss_fn(self.lstm_model(X), y)
                        loss.backward()
                        opt.step()
                    self.lstm_model.eval()
                    F_mean, F_std = rolling_forecasts_multi_mc(self.lstm_model, self.env_ref.data, self.look_back, self.k, mc_runs=10)
                    self.env_ref.forecast_mean = F_mean
                    self.env_ref.forecast_std = F_std
        return True


# --- ۵. ارزیابی و استخراج معیارها ---

def compute_forecasting_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    mask = actual > 1e-5
    act_m, pred_m = actual[mask], pred[mask]
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2) + 1e-8
    r2 = float(1.0 - (ss_res / ss_tot))
    mape = float(np.mean(np.abs((act_m - pred_m) / act_m))) if len(act_m) > 0 else 0.0
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def run_evaluation_rollout(env: EdgeEnv, policy_type: str = "drl", agent: PPO = None, forecast_matrix: np.ndarray = None):
    obs, _ = env.reset()
    done = False
    infos = []
    while not done:
        t = env.current_step
        if policy_type == "drl" and agent is not None:
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)
        elif policy_type == "rule" and forecast_matrix is not None:
            w_pred = forecast_matrix[t, 0]
            c_curr = env.allocated_resource
            action = 2 if w_pred > c_curr + 0.05 else (0 if w_pred < c_curr - 0.05 else 1)
        else:
            action = 1
        obs, reward, terminated, truncated, info = env.step(action)
        infos.append(info)
        done = terminated or truncated
    return infos


def summarize_control_performance(infos: list) -> dict:
    overs = [i["over"] for i in infos]
    unders = [i["under"] for i in infos]
    rewards = [i["reward"] for i in infos]
    sla_viols = [i["sla_violation"] for i in infos]

    tp = sum(1 for i in infos if not i["sla_violation"] and i["u_t"] > 0.1)
    fp = sum(1 for i in infos if i["over"] > 0.1)
    tn = sum(1 for i in infos if i["over"] <= 0.1 and not i["sla_violation"])
    fn = sum(1 for i in infos if i["sla_violation"])

    acc = (tp + tn) / max(1, (tp + fp + tn + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-5, (prec + rec))

    return {
        "SatisfactionRate": float(1.0 - np.mean(sla_viols)),
        "MeanOver": float(np.mean(overs)),
        "MeanUnder": float(np.mean(unders)),
        "ResourceWaste": float(np.mean(np.square(overs))),
        "LatencyPenalty": float(np.mean(np.square(unders))),
        "TotalReward": float(np.sum(rewards)),
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }


# --- ۶. اجرای اصلی پایپ‌لاین ---

def main():
    print("================================================================================")
    print("آغاز اجرای پایپ‌لاین ارزیابی کامل FGRL / HybridEdge")
    print("================================================================================")

    data_path = "dataset/azure_invocations_per_minute.csv"
    if os.path.exists(data_path) and os.path.getsize(data_path) > 100:
        df_raw = pd.read_csv(data_path)
        col = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])][-1]
        raw_series = df_raw[col].fillna(0.0).to_numpy().astype(np.float32)
    else:
        print("💡 داده خام در dataset/ یافت نشد؛ تولید داده‌های استاندارد جایگزین Azure...")
        raw_series = generate_synthetic_azure_trace(n_points=3500)

    cleaned_series = preprocess_workload_series(raw_series)
    scaler = MinMaxScaler1D()
    scaler.fit(cleaned_series)
    scaled_series = scaler.transform(cleaned_series)

    train_data, test_data = train_test_split_series(scaled_series, train_ratio=0.7)
    look_back, k = 24, 3

    print("\n[۱/۴] آموزش مدل‌های پیش‌بینی سری زمانی...")
    lstm_model = train_deep_forecaster(LSTMForecaster(look_back, k), train_data, look_back, k)
    bilstm_model = train_deep_forecaster(BiLSTMForecaster(look_back, k), train_data, look_back, k)
    gru_model = train_deep_forecaster(GRUForecaster(look_back, k), train_data, look_back, k)

    F_lstm_mean, F_lstm_std = rolling_forecasts_multi_mc(lstm_model, test_data, look_back, k)
    F_bilstm_mean, _ = rolling_forecasts_multi_mc(bilstm_model, test_data, look_back, k)
    F_gru_mean, _ = rolling_forecasts_multi_mc(gru_model, test_data, look_back, k)
    F_ma_mean, _ = moving_average_forecast(test_data, look_back, k)
    F_arima_mean, _ = arima_forecast(test_data, look_back, k)

    print("\n[۲/۴] ساخت و آموزش عامل PPO هیبریدی...")
    train_env = EdgeEnv(train_data, F_lstm_mean, F_lstm_std, look_back=look_back, k=k)
    policy_kwargs = dict(
        features_extractor_class=FusionFeaturesExtractor,
        features_extractor_kwargs=dict(look_back=look_back, k=k, features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )

    hybrid_ppo = PPO("MlpPolicy", train_env, learning_rate=3e-4, n_steps=1024, batch_size=64, policy_kwargs=policy_kwargs, verbose=0)
    hybrid_ppo.learn(total_timesteps=12000, callback=[EntropyTargetCallback(), OnlineLSTMCallback(train_env, lstm_model, look_back, k)])

    train_env_drl = EdgeEnv(train_data, np.zeros_like(F_lstm_mean), np.zeros_like(F_lstm_std), look_back=look_back, k=k)
    drl_only_ppo = PPO("MlpPolicy", train_env_drl, learning_rate=3e-4, verbose=0)
    drl_only_ppo.learn(total_timesteps=10000)

    print("\n[۳/۴] اجرای ارزیابی مقایسه‌ای بر روی داده‌های آزمون...")
    models_config = [
        ("Hybrid (LSTM+DRL)", "drl", hybrid_ppo, F_lstm_mean),
        ("DRL-Only", "drl", drl_only_ppo, F_lstm_mean),
        ("LSTM-Only (Rule)", "rule", None, F_lstm_mean),
        ("GRU-Only (Rule)", "rule", None, F_gru_mean),
        ("BiLSTM-Only (Rule)", "rule", None, F_bilstm_mean),
        ("MovingAverage", "rule", None, F_ma_mean),
        ("ARIMA-Rule", "rule", None, F_arima_mean),
    ]

    results_list = []
    for name, p_type, agent, f_mat in models_config:
        eval_env = EdgeEnv(test_data, f_mat, F_lstm_std if "Hybrid" in name else None, look_back=look_back, k=k)
        infos = run_evaluation_rollout(eval_env, policy_type=p_type, agent=agent, forecast_matrix=f_mat)
        ctrl_perf = summarize_control_performance(infos)
        actuals = test_data[look_back : look_back + len(infos)]
        preds = f_mat[look_back : look_back + len(infos), 0]
        f_perf = compute_forecasting_metrics(actuals, preds)
        results_list.append({"Model": name, **f_perf, **ctrl_perf})

    results_df = pd.DataFrame(results_list)
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/comparison_results_chapter4.csv", index=False)

    print("\n" + "=" * 80)
    print("نتایج پایپ‌لاین با موفقیت در فایل 'outputs/comparison_results_chapter4.csv' ذخیره شد.")
    print("=" * 80)

if __name__ == "__main__":
    main()
