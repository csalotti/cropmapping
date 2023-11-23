from pytorch_lightning.cli import LightningArgumentParser, LightningCLI


class SupervisedCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments("data.classes", "model.classes")
