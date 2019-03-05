import source.transfer.vgg16 as vgg
import source.transfer.resnet50 as resnet50
import source.transfer.xception as xception_net
import source.plots as plots
import source.netSet as net_set
import source.imgToArr as img_to_arr


def main():
    numbers = plots.Numbers()
    for i in range(8):
        net_data = net_set.load_net_set((i + 1) * 5)
        # results = vgg16(net_data)
        results = resnet(net_data)
        # results = xception(net_data)
        plots.add_info(results, numbers)
    plots.save_all_plots(numbers, 'resnet')



def vgg16(data):
    print("vgg16 ...")
    # dresses = img_to_arr.load_images()
    # dr = vgg.read_images(dresses)
    # vgg_dresses = vgg.predict_by_vgg16(dr)
    vgg_dresses = vgg.load_vgg_dresses()
    return vgg.do_train(vgg_dresses, data)
    print("finish vgg16 ...")


def resnet(data):
    print("start resnet ...")
    # dresses = img_to_arr.load_images()
    # dr = resnet50.read_images(dresses)
    # resnet_dresses = resnet50.predict_resnet(dr)
    resnet_dresses = resnet50.load_resnet_dresses()
    return resnet50.do_train(resnet_dresses, data)
    print("finish resnet ...")


def xception(data):
    print("start xception ...")
    # dresses = img_to_arr.load_images()
    # dr = xception_net.read_images(dresses)
    # xception_dresses = xception_net.predict_xception(dr)
    xception_dresses = xception_net.load_xception_dresses()
    return xception_net.do_train(xception_dresses, data)
    print("finish xception ...")


if __name__ == "__main__":
    main()
