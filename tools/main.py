from tangle import Config, Trainer, Inference
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train','validate','infer'], help='functions')

def main():

    args = parser.parse_args()
    
    if args.mode == 'train':
        # train the model
        cfg = Config(config_type='train')
        trainer = Trainer(config=cfg)
        trainer.train()

    elif args.mode == 'infer':
        cfg = Config(config_type='infer')
        inference = Inference(config=cfg)

        print(cfg.sepp_ckpt, cfg.sepd_ckpt)

        inference.infer()

if __name__ == '__main__':
    main()
    
