import keras_model
import numpy as np
import tensorflow as tf

def confusion_matrix_info(y_true, y_pred, labels=['normal', 'bacteria', 'viral', 'COVID-19'],
                          title='confusion matrix'):
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import confusion_matrix, f1_score
    import matplotlib.pyplot as plt
    C2 = confusion_matrix(y_true, y_pred)
    C = pd.DataFrame(C2, columns=labels, index=labels)
    m, _ = C2.shape
    for i in range(m):
        precision = C2[i, i] / sum(C2[:, i])
        recall = C2[i, i] / sum(C2[i, :])
        f1 = 2 * precision * recall / (precision + recall)
        print('In class {}:\t total samples: {}\t true predict samples: {}\t'
              'acc={:.4f},\trecall={:.4f},\tf1-score={:.4f}'.format(
            labels[i], sum(C2[i, :]), C2[i, i], precision, recall, f1))
    print('-' * 100, '\n', 'average f1={:.4f}'.format(f1_score(y_true, y_pred, average='micro')))

    f, ax = plt.subplots()
    sns.heatmap(C, annot=True, ax=ax, cmap=plt.cm.binary)
    ax.set_title(title)
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig(title+'.jpg')


def train():
    import keras
    x = np.load('data/x_train.npy')
    y = np.load('data/y_train.npy')
    y = tf.keras.utils.to_categorical(y, 4)
    model = keras_model.keras_model_build()
    # model.summary()
    opt=tf.keras.optimizers.Adam(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    hist = model.fit(x, y, batch_size=32, epochs=100, verbose=1)
    model.save('model.h5')
    y_pred = model.predict(x)
    confusion_matrix_info(np.argmax(y, axis=1), np.argmax(y_pred, axis=1),title='confusion_matrix_train')


def test():
    model = tf.keras.models.load_model('model.h5')
    xt = np.load('data/x_test.npy')
    yt = np.load('data/y_test.npy')
    y_pred = model.predict(xt)
    confusion_matrix_info(yt, np.argmax(y_pred, axis=1),title='confusion_matrix_test')


def test2():
    model = tf.keras.models.load_model('model.h5')
    xt = np.load('data/x_train.npy')
    yt = np.load('data/y_train.npy')
    y_pred = model.predict(xt)
    confusion_matrix_info(yt, np.argmax(y_pred, axis=1),title='confusion_matrix_train')




if __name__ == '__main__':
    # train()
    test2()
