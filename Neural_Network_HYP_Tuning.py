import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import
def create_model(units=units,epochs=5,batch_size=5,optimizer="adam",learning_rate=0.01,momentum=0,init_model="uniform",activation="relu",dropout_rate=0..0,weight_constraint=0):
  model=Seequential()
  model.add(Dense(units,input_dim="shape",activation=activation))
  """
  .....
  ......
  .....
  ......
  Add as many layers as you want and get grid scores for each model build
  """"
  #ADD optimizer if learning_rate and momentum is specified
  optimizer=SGD(lr=learning_rate,momentum=momentum)
  model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
  return model
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
units= [1, 5, 10, 15, 20, 25, 30]#as per req can be tuned
grid_dict=dict(batch_size=batch_size,epochs=epochs,learn_rate=learn_rate,momentum=momentum,init_mode=init_mode,activation=activation,dropout_rate=dropout_rate,units=units)
grid=GridSearchCV(estimator=model,param_grid=grid_dict,n_jobs=-1,cv=3)
grid_result=grid.fit(X_train,Y_train)
print(grid_result.best_score_)
