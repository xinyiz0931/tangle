from tangle import Config, Trainer, Inference
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train','validate','infer'], help='functions')

def main():

    args = parser.parse_args()
    config_path = "./cfg\\config.yaml"
    
    if args.mode == 'train':
        # train the model
        cfg = Config(config_path=config_path, config_type='train')
        trainer = Trainer(config=cfg)
        trainer.train()

    elif args.mode == 'infer':
        cfg = Config(config_path=config_path, config_type='infer')
        inference = Inference(config=cfg)
        inference.infer()

if __name__ == '__main__':
    main()
    
