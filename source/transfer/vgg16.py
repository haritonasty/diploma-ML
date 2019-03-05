from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Flatten, Embedding, dot
from keras.models import Model
from source.metrics import Metrics

import numpy as np
import pandas as pd
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))
num_classes = 800
num_epoch = 100
image_input = Input(shape=(224, 224, 3))


def main():
    # images = read_images()
    # vgg_dresses = predict_by_vgg16(images)
    vgg_dresses = load_vgg_dresses()
    do_train(vgg_dresses)


def read_images(dresses):
    print("start reading images ...")
    dresses = np.array(dresses)
    x = dresses.reshape(103, 224, 224, 3)
    print("finish reading images ...")
    return x


def predict_by_vgg16(imgs, n=20):
    print("start doing vgg_dresses ...")
    model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
    last_layer = model.get_layer('fc1').output
    # out = Dense(num_classes, activation='relu', name='output')(last_layer)
    custom_vgg_model = Model(image_input, last_layer)

    for layer in custom_vgg_model.layers:
        layer.trainable = False

    predict_vgg16 = custom_vgg_model.predict(imgs, batch_size=n)

    pd.DataFrame(predict_vgg16).to_csv(
        os.path.join(my_path, "../../data/marks/models/vgg/predict_vgg16.csv"), index=False)
    print("finish vgg_dresses ...")
    return predict_vgg16


def load_vgg_dresses():
    return np.array(pd.read_csv(os.path.join(my_path, "../../data/marks/models/vgg/predict_vgg16.csv")))


def do_train(dresses, net_set):
    dresses_ids_train = net_set['dresses_ids_train']
    dresses_ids_test = net_set['dresses_ids_test']
    girls_ids_train = net_set['girls_ids_train']
    girls_ids_test = net_set['girls_ids_test']
    y_true_train = net_set['y_true_train']
    y_true_test = net_set['y_true_test']

    girls_ids_train = (girls_ids_train.reshape(-1, 1))
    girls_ids_test = (girls_ids_test.reshape(-1, 1))
    y_true_train = (y_true_train.reshape(-1, 1))
    y_true_test = (y_true_test.reshape(-1, 1))

    dresses_train = []
    dresses_test = []

    print("start images ids to images train...")
    for i in dresses_ids_train:
        dresses_train.append(dresses[i])

    print("start images ids to images test...")
    for i in dresses_ids_test:
        dresses_test.append(dresses[i])

    dresses_train = np.array(dresses_train)
    dresses_test = np.array(dresses_test)
    print("finish images ids to images...")

    input_dim = 110

    user_id = Input(shape=(1,), name='user')
    dress_input = Input(shape=(4096,), name='dresses_input')
    dress_vec = Dense(num_classes, activation='relu', name='dresses_vec')(dress_input)
    user_vec = Embedding(output_dim=num_classes, input_dim=input_dim, embeddings_initializer='uniform', input_length=1, name='user_embedding')(user_id)
    user_vec = Flatten()(user_vec)
    dot_matrics = dot([user_vec, dress_vec], axes=1, name='user_dot_dress')
    output = Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(dot_matrics)
    model_collaboraty = Model(inputs=[user_id, dress_input], outputs=output)
    # model_collaboraty.summary()
    model_collaboraty.compile('adam', loss='binary_crossentropy', metrics=['acc'])

    print("start training model ...")
    metrics = Metrics()
    # predict_before_train = model_collaboraty.predict([girls_ids_test, dresses_test])
    # f1s_before, recall_before, precision_before = metrics.get_score([y_true_test, predict_before_train.round()])
    hist = model_collaboraty.fit([girls_ids_train, dresses_train], y_true_train, epochs=num_epoch)
                                 # validation_data=([girls_ids_test, dresses_test], y_true_test))
    predict_after_train = model_collaboraty.predict([girls_ids_test, dresses_test])
    f1s_after, recall_after, precision_after = metrics.get_score([y_true_test, predict_after_train.round()])
    predict_train = model_collaboraty.evaluate([girls_ids_train, dresses_train], y_true_train)
    predict_test = model_collaboraty.evaluate([girls_ids_test, dresses_test], y_true_test)

    # print(metrics.val_f1s, metrics.val_recalls, metrics.val_precisions)
    print("finish training model ...")


    return predict_train[0], predict_test[0], predict_train[1], predict_test[1], f1s_after, recall_after, precision_after


if __name__ == "__main__":
    main()
