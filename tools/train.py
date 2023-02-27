from tangle import Config, Trainer

def main():
    # train the model
    cfg = Config(config_type='train')
    trainer = Trainer(config=cfg)
    trainer.train()

if __name__ == '__main__':
    main()