import csv
from argparse import ArgumentParser

def add_dropout_arg(parser):
    parser.add_argument('--save-perplexity', type=str, default='',
                       help='path to save perplexity logs')
    return parser

def save_perplexity_log(path, epoch, train_ppl, valid_ppl, test_ppl, dropout):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if epoch == 1:  # Write header
            writer.writerow(['Epoch', 'Train_PPL', 'Valid_PPL', 'Test_PPL', 'Dropout'])
        writer.writerow([epoch, train_ppl, valid_ppl, test_ppl, dropout])

# In train() function, add:
if args.save_perplexity:
    save_perplexity_log(args.save_perplexity, epoch, train_ppl, valid_ppl, test_ppl, args.dropout)
