import argparse
from transformers import AutoTokenizer

from utils import *
from train_utils import *
from model import *
from model2 import *
from model_galore import *
from galore_optimizer import *
import torch


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_data_loader(
        'data/in_domain_train.tsv', args.batch_size, tokenizer)
    val_loader = get_data_loader(
        'data/in_domain_dev.tsv', args.batch_size, tokenizer, shuffle=False)
    

    #Memory Calculation  
    # torch.cuda.reset_max_memory_allocated(args.device)
    start_memory = torch.cuda.max_memory_allocated(args.device)
    print(f"Memory before training: {start_memory / 1e9:.2f}GB")

    if args.mode == "gen":
        model = GPT(args.gpt_variant, is_gen=True).to(args.device)
        model.eval()

        # TODO: You can add your super creative prompt here
        prompt = "My name is Inigo Montoya. You killed my father. Prepare to die."

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print("", tokenizer.decode(output[0]), sep="\n")

    elif args.mode == "classify":    
        criterion = torch.nn.CrossEntropyLoss()

        model = GPTO(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        loss, acc = evaluate(model, val_loader, criterion=criterion, device='cuda:4')
        print(f"Initial Loss: {loss}, Accuracy: {acc}")

        model2 = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        model2.load_trainable_params(args.model_path)
        loss2, acc2 = evaluate(model2, val_loader, criterion=criterion, device='cuda:4')
        print(f"Loaded Loss: {loss2}, Accuracy: {acc2}")

    elif args.mode == "LoRA":    
        model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        train(model, train_loader, val_loader, optimizer, criterion, args.device, args.epochs)

        model.save_trainable_params(args.model_path)
    
    elif args.mode == "galore":  

        model = GPT_galore(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        optimizer = G_ADAM(model.parameters(), lr=args.lr, rank=args.LoRA_rank, update_interval=50)
        criterion = torch.nn.CrossEntropyLoss()
        train(model, train_loader, val_loader, optimizer, criterion, args.device, args.epochs)
        model.save_trainable_params('models/galore.pth')
    
    elif args.mode == "gpt_trainable":  

        model = GPT_galore(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        train(model, train_loader, val_loader, optimizer, criterion, args.device, args.epochs)
        model.save_trainable_params('models/galore.pth')
        
    elif args.mode == "distil":
        teacher_model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        teacher_model.load_trainable_params(args.model_path)
        teacher_model.eval()

        student_model = DistilRNN(hidden_dim=args.hidden_dim).to(args.device) 

        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
        distil_criterion = torch.nn.KLDivLoss()
        criterion = torch.nn.CrossEntropyLoss()
        train_distil(teacher_model, student_model, train_loader, val_loader, optimizer, criterion, distil_criterion, args.device, args.epochs)
    elif args.mode == "rnn":
        model = DistilRNN().to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        train(model, train_loader, val_loader, optimizer, criterion, args.device, args.epochs, is_rnn=True)

    else:
        print("Invalid mode")
        return
    

    #Memory Calculation
    end_memory = torch.cuda.max_memory_allocated(args.device)
    print(f"Memory after training: {end_memory / 1e9:.2f}GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2")
    parser.add_argument("mode", type=str, choices=["gen", "LoRA", "distil", "rnn", "classify", 'galore', 'gpt_trainable'], help="Mode to run the program in")
    parser.add_argument("sr_no", type=int, help="5 digit SR number")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--gpt_variant", type=str, default="gpt2-medium", choices=["gpt2", "gpt2-medium"], help="Model to use")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--model_path", type=str, default="models/LoRA.pth", help="Path to save the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--LoRA_rank", type=int, default=4, help="Low rank matrix bottleneck")
    parser.add_argument("--hidden_dim", type=int, default=256, help="RNN Hidden Dimensions")
    # TODO: Add more arguments as needed
    
    args = parser.parse_args()
    args.device = torch.device(
        "cuda:4" if torch.cuda.is_available() and args.gpu_id >= 0 else\
        "mps" if torch.backends.mps.is_available() else "cpu")
    
    seed_everything(args.sr_no)

    main(args)
