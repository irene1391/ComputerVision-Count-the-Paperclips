from collections import OrderedDict
from tqdm import tqdm
from dataset import get_dataloader
from config import get_config
from agent import get_agent
import csv
import torch
import numpy as np

def main():
    test_data = True
    pretrain = True

    # create experiment config
    config = get_config()

    # create network and training agent
    tr_agent = get_agent(config)
    print(tr_agent.net)

    # load from checkpoint if provided
    if pretrain:
        tr_agent.load_ckpt("latest")

    # create dataloader
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'validation')
    test_loader = get_dataloader(config, 'test')

    # start training
    clock = tr_agent.clock

    # test
    if test_data == True:
        pbar = tqdm(test_loader)
        writer = csv.writer(open("../result.csv", "w"))
        writer.writerow(["id", "clip_count"])
        for b, data in enumerate(pbar):
            outputs, losses = tr_agent.val_func(data[0].cuda(), data[1].cuda())
            outputs = outputs.argmax().cpu().numpy()
            writer.writerow([b+25001, outputs])

    for e in range(clock.epoch, config.epochs):
        if e % config.val_frequency == 0:
            loss = 0
            for b, data in enumerate(val_loader):
                outputs, losses = tr_agent.val_func(data[0].cuda(), data[1].cuda())
                loss += losses['loss']
            loss /= len(val_loader)
            print("EPOCH {} valid loss : {}".format(e, loss))
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = tr_agent.train_func(data[0].cuda(), data[1].cuda())

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            clock.tick()

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
