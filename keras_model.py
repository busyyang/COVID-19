import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, add, Flatten, Dense


def PEPXModel(input_tensor, filters, name):
    x = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', name=name + 'FP')(input_tensor)
    x = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', name=name + 'Expansion')(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', name=name + 'DWConv3_3')(x)
    x = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', name=name + 'SP')(x)
    x = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', name=name + 'Extension')(x)
    return x


def keras_model_build(input_size=(224, 224, 3)):
    # 输入
    input = Input(shape=input_size, name='input')
    x = Conv2D(input_shape=input_size, filters=64, kernel_size=(7, 7), activation='relu', padding='same',
               strides=(2, 2))(input)
    x = MaxPool2D(pool_size=(2, 2))(x)
    # Stage1结构
    y_1_1 = PEPXModel(x, 256, 'PEPX1.1')
    y_1_2 = PEPXModel(y_1_1, 256, 'PEPX1.2')
    y_1_3 = PEPXModel(y_1_2, 256, 'PEPX1.3')
    y_1_3 = keras.layers.add([y_1_1, y_1_3])
    # Stage2结构
    y_2_1 = PEPXModel(add([y_1_3, y_1_2, y_1_1]), 512, 'PEPX2.1')
    y_2_1 = MaxPool2D(pool_size=(2, 2))(y_2_1)
    y_2_2 = PEPXModel(y_2_1, 512, 'PEPX2.2')
    y_2_3 = PEPXModel(add([y_2_1, y_2_2]), 512, 'PEPX2.3')
    y_2_4 = PEPXModel(add([y_2_1, y_2_2, y_2_3]), 512, 'PEPX2.4')
    # Stage3结构
    y_3_1 = PEPXModel(add([y_2_1, y_2_2, y_2_3, y_2_4]), 1024, 'PEPX3.1')
    y_3_1 = MaxPool2D(pool_size=(2, 2))(y_3_1)
    y_3_2 = PEPXModel(y_3_1, 1024, 'PEPX3.2')
    y_3_3 = PEPXModel(add([y_3_1, y_3_2]), 1024, 'PEPX3.3')
    y_3_4 = PEPXModel(add([y_3_1, y_3_2, y_3_3]), 1024, 'PEPX3.4')
    y_3_5 = PEPXModel(add([y_3_1, y_3_2, y_3_3, y_3_4]), 1024, 'PEPX3.5')
    y_3_6 = PEPXModel(add([y_3_1, y_3_2, y_3_3, y_3_4, y_3_5]), 1024, 'PEPX3.6')
    # Stage4结构
    y_4_1 = PEPXModel(add([y_3_1, y_3_2, y_3_3, y_3_4, y_3_5, y_3_6]), 2048, 'PEPX4.1')
    y_4_1 = MaxPool2D(pool_size=(2, 2))(y_4_1)
    y_4_2 = PEPXModel(y_4_1, 2048, 'PEPX4.2')
    y_4_3 = PEPXModel(add([y_4_1, y_4_2]), 2048, 'PEPX4.3')
    # FC
    fla = Flatten()(add([y_4_1, y_4_2, y_4_3]))
    d1 = Dense(1024, activation='relu')(fla)
    d2 = Dense(256, activation='relu')(d1)
    output = Dense(4, activation='softmax')(d2)

    return keras.models.Model(input, output)


if __name__ == '__main__':
    model = keras_model_build()
    model.summary()
