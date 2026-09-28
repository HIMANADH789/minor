import torch

def main():
    from rich.console import Console
    console = Console()
    console.print("[bold green]=== Environment Verification ===[/bold green]")
    console.print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        console.print("[bold cyan]CUDA: Available[/bold cyan]")
        console.print(f"CUDA Version: {torch.version.cuda}")
        console.print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        console.print(f"BF16 Support: {torch.cuda.is_bf16_supported()}")
        console.print(f"TF32 Matmul Allow: {torch.backends.cuda.matmul.allow_tf32}")
        console.print(f"TF32 CuDNN Allow: {torch.backends.cudnn.allow_tf32}")
    else:
        console.print("[bold red]CUDA: NOT Available[/bold red]")

if __name__ == "__main__":
    main()
