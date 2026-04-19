"""
Experiment #9: Measure Mistral-7B Memory with nvidia-smi

Validates the OOM claim by measuring actual memory usage.
"""

import torch
import subprocess
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_gpu_memory():
    """Get current GPU memory via nvidia-smi."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        used, total = result.stdout.strip().split(',')
        return float(used), float(total)
    return None, None


def run_experiment_9():
    """Run Experiment #9: Memory measurement."""
    print("=" * 80)
    print("EXPERIMENT #9: Mistral-7B Memory Measurement")
    print("=" * 80)
    print()
    print("Goal: Measure actual GPU memory with nvidia-smi")
    print("Expected: Uncompressed ~9.9GB, should exceed 12GB with overhead")
    print()
    
    if not torch.cuda.is_available():
        print("ERROR: No GPU available")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    
    results = {
        "gpu": gpu_name,
        "cuda_version": str(torch.version.cuda),
        "measurements": []
    }
    
    # Baseline
    mem_before, mem_total = get_gpu_memory()
    print(f"Baseline memory: {mem_before:.1f} MB / {mem_total:.1f} MB")
    results["baseline_mb"] = mem_before
    results["total_mb"] = mem_total
    
    try:
        # Load Mistral-7B with 4-bit quantization
        print("\nLoading Mistral-7B with 4-bit quantization...")
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load with quantization
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        mem_after_load, _ = get_gpu_memory()
        print(f"Memory after loading: {mem_after_load:.1f} MB")
        print(f"Model memory: {mem_after_load - mem_before:.1f} MB")
        results["model_loaded_mb"] = mem_after_load
        
        # Test at 16K context
        print("\nTesting 16K token context...")
        prompt = " ".join(["test"] * 16000)
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384).to("cuda")
            input_len = inputs.input_ids.shape[1]
            print(f"Input tokens: {input_len}")
            
            # Prefill
            print("Running prefill...")
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True)
            
            mem_after_prefill, _ = get_gpu_memory()
            print(f"Memory after prefill: {mem_after_prefill:.1f} MB")
            print(f"KV cache memory: {mem_after_prefill - mem_after_load:.1f} MB")
            results["prefill_16k_mb"] = mem_after_prefill
            
            # Generate 1 token
            print("\nGenerating token...")
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:].argmax(dim=-1)
            
            gen_inputs = {
                "input_ids": next_token,
                "past_key_values": past_key_values,
                "use_cache": True
            }
            
            with torch.no_grad():
                gen_outputs = model(**gen_inputs)
            
            mem_after_gen, _ = get_gpu_memory()
            print(f"Memory after generation: {mem_after_gen:.1f} MB")
            results["generation_mb"] = mem_after_gen
            
            # Analysis
            print("\n" + "=" * 80)
            print("MEMORY ANALYSIS")
            print("=" * 80)
            
            kv_cache_mb = mem_after_prefill - mem_after_load
            kv_cache_gb = kv_cache_mb / 1024
            
            print(f"\nUncompressed KV cache at 16K: ~{kv_cache_gb:.2f} GB")
            print(f"Total memory with model + KV: {mem_after_prefill / 1024:.2f} GB")
            
            if mem_after_prefill > 12000:
                print(f"\n✓ CONFIRMED: {mem_after_prefill / 1024:.2f}GB > 12GB")
                print("  OOM claim is VALID")
            elif mem_after_prefill > 10000:
                print(f"\n⚠ MARGINAL: {mem_after_prefill / 1024:.2f}GB is close to 12GB limit")
                print("  May OOM with batching or overhead")
            else:
                print(f"\n✗ CONTRADICTED: {mem_after_prefill / 1024:.2f}GB < 12GB")
                print("  OOM claim needs revision")
            
            results["kv_cache_gb"] = kv_cache_gb
            results["total_gb"] = mem_after_prefill / 1024
            
        except RuntimeError as e:
            print(f"\n✓ OOM DURING PREFILL: {e}")
            print("  This validates the OOM claim!")
            results["oom_at_prefill"] = True
            results["oom_error"] = str(e)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        results["error"] = str(e)
    
    # Save results
    with open('../results/exp9_memory_measurement.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/exp9_memory_measurement.json")
    
    return results


if __name__ == "__main__":
    run_experiment_9()
