from get_x_get_y import get_x_get_y
from sklearn.model_selection import train_test_split

def get_splited_data(path):
    x,y=get_x_get_y(path)
    X,x_test,Y,y_test=train_test_split(x,y,test_size=.25,random_state=100)
    x_train,x_val,y_train,y_val=train_test_split(X,Y,test_size=.25,random_state=100)
    return x_train,x_val,x_test,y_train,y_val,y_test

