from tangle import Config, Inference

def main():
    # infer 
    cfg = Config(config_type='infer')
    cfg.display()
    inference = Inference(config=cfg)
    
    import os
    # it can infer a single image or all images under a folder 
    # input_path = os.path.join(cfg.root_dir, "samples/000000.png")
    input_path = os.path.join(cfg.root_dir, "samples")
    
    # output = inference.infer(data_dir=input_path, net_type="pick", show=True)
    output = inference.infer(data_dir=input_path, net_type="auto", show=True)

if __name__ == '__main__':
    main()