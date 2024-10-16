from tangle import Config, Trainer

def main():
    # train the model
    cfg = Config(config_type='train', config_file='config_train_picknet.yaml')
    # cfg = Config(config_type='train', config_file='config_train_pullnet.yaml')
    trainer = Trainer(config=cfg)
    trainer.train()

if __name__ == '__main__':
    main()