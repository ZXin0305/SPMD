import Config.config as conf
from trainer import Trainer


def main():
    cnf = conf.set_param()
    trainer = Trainer(cnf)
    trainer.run()

if __name__ == '__main__':
    main()