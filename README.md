# SVM
This project is to use SVM to predict stock price. Factor: ema, stddev, slope, rsi, wr.
The result is ok.

ps. Now I find that I don't need KFold, I just need train_test_split from sklearn.model_selection:

```
cv_score = model_selection.cross_val_score(model_list[i], X_train, y_train, cv=5)
```
