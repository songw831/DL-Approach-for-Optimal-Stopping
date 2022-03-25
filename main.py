import model


config = model.Config(t=0, x=100, K=100, r=0.02, T=1.0, delta=0.3, M=1000)
train_data = model.loadData(config)

if __name__ == '__main__':

    print(model.predict(config, train_data))

