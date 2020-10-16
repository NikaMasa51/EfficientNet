from efficientnet import EfficientNetB0
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from data_loader import DataLoader
from adam_lr_mult import Adam_lr_mult

def prepare_new_model(input_shape, class_count):
    # 学習済みモデルの取り出し
    feature_extractor = EfficientNetB0(input_shape=input_shape, weights='imagenet', include_top=False)
    # 犬猫分類器を引っ付ける
    x = feature_extractor.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(class_count, activation='sigmoid')(x)
    # 新たなモデルの定義
    model = Model(inputs=feature_extractor.input, outputs=x)
    print(model.summary())
    return model

def get_adam_for_fine_tuning(lr, decay, multiplier, model):
    lr_multiplier = {}
    # 自分が引っ付けたレイヤーの学習係数は1、学習済みの部分は小さな値を設定する
    for layer in model.layers:
        if 'dense' in layer.name:
            lr_multiplier[layer.name] = 1.0
        else:
            lr_multiplier[layer.name] = multiplier
    return Adam_lr_mult(lr=lr, decay=decay, multipliers=lr_multiplier)

def train(epochs, batch_size, input_shape, class_count):
    # 学習用画像データローダー
    train_data_loader = DataLoader('train', batch_size, input_shape, do_augmentation=True)
    train_generator = train_data_loader.get_data_loader()
    # 検証用画像データローダー
    val_data_loader = DataLoader('val', batch_size, input_shape, do_augmentation=False)
    val_generator = val_data_loader.get_data_loader()
    # モデルの生成
    model = prepare_new_model(input_shape, class_count)
    # ファインチューニング用Adamオプティマイザ
    optimizer = get_adam_for_fine_tuning(lr=1e-3, decay=1e-5, multiplier=0.01, model=model)
    # コンパイルして
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # fit_generatorするだけ
    h = model.fit_generator(train_generator, train_data_loader.iterations, epochs,
                                  validation_data=val_generator, validation_steps=val_data_loader.iterations)
    # 学習データ、検証データのロスとAccをファイルに出力
    with open('loss.csv', 'a') as f:
        for loss_t, acc_t, loss_v, acc_v in zip(h.history['loss'], h.history['acc'], h.history['val_loss'], h.history['val_acc']):
            f.write(str(loss_t) + ',' + str(acc_t) + ',' + str(loss_v) + ',' + str(acc_v) + '\n')


if __name__ == '__main__':
    epochs = 100
    batch_size = 16
    input_shape = (224, 224, 3)
    class_count = 2
    train(epochs, batch_size, input_shape, class_count)
