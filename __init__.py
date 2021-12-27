import model_manager
import tests
from os import listdir

models = []

for i in listdir("StockCSVs"):
    m = model_manager.PureHistoryLSTM("No Stockname", i.strip(".csv"), 30)
    models.append(m)
    m.train(50, 11)
    m.save()
    x, y = m.x_test, m.y_test

    t1 = tests.OpenCloseDeltaTrade(m, x, y)
    t1.test_function()
    t1.print_res()
