# FGRL: Forecast-Guided Reinforcement Learning for Serverless Resource Management

یک چارچوب هوشمند، تطبیقی و آگاه از عدم‌قطعیت جهت مدیریت منابع در محیط‌های رایانش لبه بدون سرور (Serverless Edge Computing) مبتنی بر تلفیق پیش‌بینی چندگامی سری زمانی، برآورد عدم‌قطعیت (Monte-Carlo Dropout) و یادگیری تقویتی عمیق (PPO).

## 📌 ویژگی‌های پروژه
- **پیش‌بینی چندگامی بار کاری**: پیاده‌سازی مدل‌های عصبی بازگشتی LSTM, Bi-LSTM و GRU.
- **تخمین عدم‌قطعیت**: مدل‌سازی عدم‌قطعیت پیش‌بینی با الگوریتم Monte-Carlo Dropout.
- **تشخیص Concept Drift**: پایش برخط تغییر توزیع بار کاری با آزمون آماری Page-Hinkley.
- **استخراج ویژگی چندوجهی**: ماژول FusionFeaturesExtractor برای شبکه Actor-Critic.
- **ریزآموزی برخط**: ماژول OnlineLSTMCallback برای بازآموزی پیش‌بینی هنگام بروز تغییر رژیم.

## 🛠️ نحوه اجرا
```bash
# ۱. نصب کتابخانه‌های پیش‌نیاز
pip install -r requirements.txt

# ۲. اجرای پایپ‌لاین مدل‌سازی و ساخت جداول ارزیابی
python main_fgrl_pipeline.py

# ۳. رسم و ذخیره تمامی ۱۸ شکل و نمودار فصل ۴
python generate_chapter4_plots.py

