import click
import logging
import os
import pathlib
import sys
#
from training import run_train


# Add the directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# log
logging.basicConfig(filename=str(pathlib.Path(__file__).parents[0].joinpath('text-coding.log')),
                    level=logging.DEBUG)
                    
                    
@click.command()
@click.argument('target_cat', default='y')
@click.argument('data_dir', type=click.Path(exists=True)) #Path to data directory
@click.argument('model_name', default="SVC") #Name of the model: e.g., SVM, CNB ...
#
@click.option('--train_cat', help='Train categories', default=False, is_flag=True)
@click.option('--balanced', help='Balanced corpus', default=False, is_flag=True)
def main_task(train_cat, data_dir, target_cat, model_name, balanced):
    print(data_dir, target_cat, model_name, balanced)
    if train_cat:
        x = 'x'
        print(run_train(data_dir, x, target_cat, train_model=True, bal=balanced, model_target=model_name)) 
    else:
        click.UsageError('Illegal user: Please indicate a running option. ' \
                         'Type --help for more information of the available ' \
                         'options.')


if __name__ == '__main__':
    cd_name = os.path.basename(os.getcwd())
    if cd_name != 'src':
        click.UsageError('Illegal use: This script must run from the src directory')
    else:
        main_task()

