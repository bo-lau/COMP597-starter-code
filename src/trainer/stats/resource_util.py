import logging
import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils
import torch
import pynvml
import psutil
import os
import time
import csv

logger = logging.getLogger(__name__)

trainer_stats_name = "resource_util"

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    """Construct a ResourceUtilStats instance from configuration.
    
    Parameters
    ----------
    conf
        Configuration object
    **kwargs
        Additional keyword arguments, should include 'device'
        
    Returns
    -------
    ResourceUtilStats
        Initialized resource utilization statistics tracker
    """
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to resource_util trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    
    # Get output directory from config if available
    output_dir = "."
    try:
        if hasattr(conf.trainer_stats_configs, 'resource_util'):
            output_dir = conf.trainer_stats_configs.resource_util.output_dir
    except AttributeError:
        pass
    
    return ResourceUtilStats(device=device, output_dir=output_dir)

class ResourceUtilStats(base.TrainerStats):
    """Tracks GPU utilization, memory consumption, and I/O statistics.
    
    This class measures:
    - GPU utilization (%)
    - GPU memory usage (MB)
    - CPU utilization (%), this process only (``psutil.Process().cpu_percent()``; sum across cores, can exceed 100)
    - CPU memory usage (MB)
    - Disk I/O (read/write bytes)
    
    Parameters
    ----------
    device
        The PyTorch device used for training. Used to synchronize CUDA operations
        and identify the GPU index for monitoring.
    output_dir
        Directory where statistics CSV file will be saved.
        
    Attributes
    ----------
    device : torch.device
        The PyTorch device as provided to the constructor.
    gpu_util_stats : RunningStat
        Statistics for GPU utilization percentage.
    gpu_memory_stats : RunningStat
        Statistics for GPU memory usage in bytes.
    cpu_util_stats : RunningStat
        Statistics for CPU utilization (stored as ``int(round(percent * 100))``, same scale as GPU util stats).
    cpu_memory_stats : RunningStat
        Statistics for CPU memory usage in bytes.
    disk_read_stats : RunningStat
        Statistics for disk read operations in bytes.
    disk_write_stats : RunningStat
        Statistics for disk write operations in bytes.
    """
    
    def __init__(self, device: torch.device, output_dir: str = ".") -> None:
        super().__init__()
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize NVML for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_index = device.index if device.type == 'cuda' else 0
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.gpu_available = True
        except Exception as e:
            logger.warning(f"Failed to initialize GPU monitoring: {e}")
            self.gpu_available = False
            self.gpu_handle = None
        
        # Statistics trackers
        self.gpu_util_stats = utils.RunningStat()  # GPU compute utilization (%)
        self.gpu_memory_util_stats = utils.RunningStat()  # GPU memory utilization (%) from NVML
        self.gpu_power_stats = utils.RunningStat()  # GPU power consumption (mW)
        self.gpu_memory_allocated_stats = utils.RunningStat()  # Memory allocated by PyTorch (actual tensor memory)
        self.gpu_memory_reserved_stats = utils.RunningStat()  # Memory reserved by PyTorch (allocator cache)
        self.cpu_util_stats = utils.RunningStat()  # Process CPU % (psutil); stored as int(percent * 100)
        self.cpu_memory_stats = utils.RunningStat()
        self.disk_read_stats = utils.RunningStat()
        self.disk_write_stats = utils.RunningStat()
        
        # Process for I/O tracking
        self.process = psutil.Process(os.getpid())
        
        # Initial I/O counters
        try:
            self._initial_io = self.process.io_counters()
        except Exception as e:
            logger.warning(f"Failed to initialize I/O tracking: {e}")
            self._initial_io = None
        
    def _get_gpu_utilization(self) -> float:
        """Get current GPU compute utilization percentage.
        
        Returns
        -------
        float
            GPU utilization percentage (0-100)
        """
        if not self.gpu_available or self.device.type != 'cuda':
            return 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            return util.gpu
        except Exception as e:
            logger.debug(f"Failed to get GPU utilization: {e}")
            return 0.0
    
    def _get_gpu_memory_utilization(self) -> float:
        """Get current GPU memory utilization percentage from NVML.
        
        Returns
        -------
        float
            GPU memory utilization percentage (0-100)
        """
        if not self.gpu_available or self.device.type != 'cuda':
            return 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            return util.memory
        except Exception as e:
            logger.debug(f"Failed to get GPU memory utilization: {e}")
            return 0.0
    
    def _get_gpu_power(self) -> int:
        """Get current GPU power consumption in milliwatts.
        
        Returns
        -------
        int
            GPU power consumption in milliwatts
        """
        if not self.gpu_available or self.device.type != 'cuda':
            return 0
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
            return power
        except Exception as e:
            logger.debug(f"Failed to get GPU power: {e}")
            return 0
    
    def _get_gpu_memory_allocated(self) -> int:
        """Get current GPU memory allocated by PyTorch (actual tensor memory) in bytes.
        
        Returns
        -------
        int
            GPU memory allocated in bytes
        """
        if self.device.type != 'cuda':
            return 0
        try:
            return torch.cuda.memory_allocated(self.device)
        except Exception as e:
            logger.debug(f"Failed to get GPU memory allocated: {e}")
            return 0
    
    def _get_gpu_memory_reserved(self) -> int:
        """Get current GPU memory reserved by PyTorch (allocator cache) in bytes.
        
        Returns
        -------
        int
            GPU memory reserved in bytes
        """
        if self.device.type != 'cuda':
            return 0
        try:
            return torch.cuda.memory_reserved(self.device)
        except Exception as e:
            logger.debug(f"Failed to get GPU memory reserved: {e}")
            return 0
    
    def _get_cpu_memory(self) -> int:
        """Get current CPU memory usage in bytes.
        
        Returns
        -------
        int
            CPU memory usage in bytes
        """
        try:
            return self.process.memory_info().rss
        except Exception as e:
            logger.debug(f"Failed to get CPU memory: {e}")
            return 0
    
    def _get_disk_io(self) -> tuple:
        """Get cumulative disk I/O in bytes (read, write).
        
        Returns
        -------
        tuple
            (read_bytes, write_bytes) since training started
        """
        if self._initial_io is None:
            return (0, 0)
        try:
            io = self.process.io_counters()
            return (io.read_bytes - self._initial_io.read_bytes, 
                    io.write_bytes - self._initial_io.write_bytes)
        except Exception as e:
            logger.debug(f"Failed to get disk I/O: {e}")
            return (0, 0)
    
    def start_train(self) -> None:
        """Called when training starts."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.process.cpu_percent(interval=None)
        logger.info("Resource utilization tracking started")
    
    def stop_train(self) -> None:
        """Called when training stops."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self._save_stats()
        logger.info("Resource utilization tracking stopped")
    
    def start_step(self) -> None:
        """Called at the start of each training step."""
        pass
    
    def stop_step(self) -> None:
        """Called at the end of each training step."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self._record_stats()
    
    def start_forward(self) -> None:
        """Called at the start of forward pass."""
        pass
    
    def stop_forward(self) -> None:
        """Called at the end of forward pass."""
        pass
    
    def start_backward(self) -> None:
        """Called at the start of backward pass."""
        pass
    
    def stop_backward(self) -> None:
        """Called at the end of backward pass."""
        pass
    
    def start_optimizer_step(self) -> None:
        """Called at the start of optimizer step."""
        pass
    
    def stop_optimizer_step(self) -> None:
        """Called at the end of optimizer step."""
        pass
    
    def start_save_checkpoint(self) -> None:
        """Called when checkpointing starts."""
        pass
    
    def stop_save_checkpoint(self) -> None:
        """Called when checkpointing stops."""
        pass
    
    def _record_stats(self) -> None:
        """Record current resource utilization statistics."""
        if self.device.type == 'cuda' and self.gpu_available:
            # GPU compute utilization (%)
            gpu_util = self._get_gpu_utilization()
            self.gpu_util_stats.update(int(gpu_util * 100))  # Store as integer percentage
            
            # GPU memory utilization (%) from NVML
            gpu_mem_util = self._get_gpu_memory_utilization()
            self.gpu_memory_util_stats.update(int(gpu_mem_util * 100))
            
            # GPU power consumption (mW)
            gpu_power = self._get_gpu_power()
            self.gpu_power_stats.update(gpu_power)
            
            # GPU memory allocated (bytes) - memory actually used by tensors
            gpu_mem_allocated = self._get_gpu_memory_allocated()
            self.gpu_memory_allocated_stats.update(gpu_mem_allocated)
            
            # GPU memory reserved (bytes) - memory reserved by PyTorch allocator
            gpu_mem_reserved = self._get_gpu_memory_reserved()
            self.gpu_memory_reserved_stats.update(gpu_mem_reserved)
        
        try:
            cpu_util = float(self.process.cpu_percent(interval=None))
        except Exception as e:
            logger.debug(f"Failed to get CPU utilization: {e}")
            cpu_util = 0.0
        self.cpu_util_stats.update(int(round(cpu_util * 100)))

        # CPU memory (bytes)
        cpu_mem = self._get_cpu_memory()
        self.cpu_memory_stats.update(cpu_mem)
        
        # Disk I/O (bytes)
        disk_read, disk_write = self._get_disk_io()
        self.disk_read_stats.update(disk_read)
        self.disk_write_stats.update(disk_write)
    
    def log_step(self) -> None:
        """Log statistics for the previous step."""
        output_parts = []
        
        if self.device.type == 'cuda' and self.gpu_available:
            gpu_util = self.gpu_util_stats.get_last() / 100.0 if self.gpu_util_stats.get_last() >= 0 else 0.0
            gpu_mem_util = self.gpu_memory_util_stats.get_last() / 100.0 if self.gpu_memory_util_stats.get_last() >= 0 else 0.0
            gpu_power_w = self.gpu_power_stats.get_last() / 1000.0 if self.gpu_power_stats.get_last() >= 0 else 0.0
            gpu_mem_allocated_mb = self.gpu_memory_allocated_stats.get_last() / (1024 * 1024) if self.gpu_memory_allocated_stats.get_last() >= 0 else 0.0
            gpu_mem_reserved_mb = self.gpu_memory_reserved_stats.get_last() / (1024 * 1024) if self.gpu_memory_reserved_stats.get_last() >= 0 else 0.0
            output_parts.append(f"GPU Util: {gpu_util:.1f}% | GPU Mem Util: {gpu_mem_util:.1f}% | GPU Power: {gpu_power_w:.1f}W | GPU Mem Alloc: {gpu_mem_allocated_mb:.1f} MB | GPU Mem Reserved: {gpu_mem_reserved_mb:.1f} MB")
        
        cpu_util_pct = (
            self.cpu_util_stats.get_last() / 100.0
            if len(self.cpu_util_stats.history) > 0 and self.cpu_util_stats.get_last() >= 0
            else 0.0
        )
        cpu_mem_mb = self.cpu_memory_stats.get_last() / (1024 * 1024) if self.cpu_memory_stats.get_last() >= 0 else 0.0
        disk_read_mb = self.disk_read_stats.get_last() / (1024 * 1024) if self.disk_read_stats.get_last() >= 0 else 0.0
        disk_write_mb = self.disk_write_stats.get_last() / (1024 * 1024) if self.disk_write_stats.get_last() >= 0 else 0.0

        output_parts.append(
            f"CPU Util: {cpu_util_pct:.1f}% | CPU Mem: {cpu_mem_mb:.1f} MB | Disk R: {disk_read_mb:.1f} MB | Disk W: {disk_write_mb:.1f} MB"
        )
        print(" | ".join(output_parts))
    
    def log_stats(self) -> None:
        """Log summary statistics."""
        if self.device.type == 'cuda' and self.gpu_available:
            print("###############   GPU COMPUTE UTILIZATION   ###############")
            self._log_stat_percentage(self.gpu_util_stats, "GPU Compute Utilization (%)")
            
            print("###############   GPU MEMORY UTILIZATION   ###############")
            self._log_stat_percentage(self.gpu_memory_util_stats, "GPU Memory Utilization (%)")
            
            print("###############   GPU POWER CONSUMPTION (W)    ###############")
            self._log_stat_watts(self.gpu_power_stats, "GPU Power")
            
            print("###############   GPU MEMORY ALLOCATED (MB)    ###############")
            self._log_stat_mb(self.gpu_memory_allocated_stats, "GPU Memory Allocated")
            
            print("###############   GPU MEMORY RESERVED (MB)    ###############")
            self._log_stat_mb(self.gpu_memory_reserved_stats, "GPU Memory Reserved")

        if len(self.cpu_util_stats.history) > 0:
            print("###############   CPU UTILIZATION (process %)    ###############")
            self._log_stat_percentage(self.cpu_util_stats, "CPU Utilization (%)")

        print("###############   CPU MEMORY (MB)    ###############")
        self._log_stat_mb(self.cpu_memory_stats, "CPU Memory")
        
        print("###############   DISK READ (MB)     ###############")
        self._log_stat_mb(self.disk_read_stats, "Disk Read")
        
        print("###############   DISK WRITE (MB)    ###############")
        self._log_stat_mb(self.disk_write_stats, "Disk Write")
    
    def _log_stat_mb(self, stat: utils.RunningStat, name: str) -> None:
        """Helper to log statistics in MB."""
        if len(stat.history) == 0:
            print(f"No data collected for {name}")
            return
        data = torch.tensor(stat.history)
        data_mb = data.to(torch.float) / (1024 * 1024)
        print(f"mean   : {data_mb.mean():.2f}")
        print(f"q0.25  : {data_mb.quantile(torch.tensor(0.25)):.2f}")
        print(f"q0.5   : {data_mb.quantile(torch.tensor(0.5)):.2f}")
        print(f"q0.75  : {data_mb.quantile(torch.tensor(0.75)):.2f}")
        print(f"max    : {data_mb.max():.2f}")
    
    def _log_stat_percentage(self, stat: utils.RunningStat, name: str) -> None:
        """Helper to log percentage statistics."""
        if len(stat.history) == 0:
            print(f"No data collected for {name}")
            return
        data = torch.tensor(stat.history)
        data_pct = data.to(torch.float) / 100.0
        print(f"mean   : {data_pct.mean():.2f}%")
        print(f"q0.25  : {data_pct.quantile(torch.tensor(0.25)):.2f}%")
        print(f"q0.5   : {data_pct.quantile(torch.tensor(0.5)):.2f}%")
        print(f"q0.75  : {data_pct.quantile(torch.tensor(0.75)):.2f}%")
        print(f"max    : {data_pct.max():.2f}%")
    
    def _save_stats(self) -> None:
        """Save summary statistics to TXT file and per-step data to CSV."""
        # Save summary statistics to TXT file
        txt_file = os.path.join(self.output_dir, 'resource_utilization_summary.txt')
        with open(txt_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("RESOURCE UTILIZATION SUMMARY STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            
            if self.device.type == 'cuda' and self.gpu_available:
                f.write("###############   GPU COMPUTE UTILIZATION   ###############\n")
                self._write_stat_percentage(f, self.gpu_util_stats, "GPU Compute Utilization (%)")
                f.write("\n")
                
                f.write("###############   GPU MEMORY UTILIZATION   ###############\n")
                self._write_stat_percentage(f, self.gpu_memory_util_stats, "GPU Memory Utilization (%)")
                f.write("\n")
                
                f.write("###############   GPU POWER CONSUMPTION (W)    ###############\n")
                self._write_stat_watts(f, self.gpu_power_stats, "GPU Power")
                f.write("\n")
                
                f.write("###############   GPU MEMORY ALLOCATED (MB)    ###############\n")
                self._write_stat_mb(f, self.gpu_memory_allocated_stats, "GPU Memory Allocated")
                f.write("\n")
                
                f.write("###############   GPU MEMORY RESERVED (MB)    ###############\n")
                self._write_stat_mb(f, self.gpu_memory_reserved_stats, "GPU Memory Reserved")
                f.write("\n")

            if len(self.cpu_util_stats.history) > 0:
                f.write("###############   CPU UTILIZATION (process %)    ###############\n")
                self._write_stat_percentage(f, self.cpu_util_stats, "CPU Utilization (%)")
                f.write("\n")

            f.write("###############   CPU MEMORY (MB)    ###############\n")
            self._write_stat_mb(f, self.cpu_memory_stats, "CPU Memory")
            f.write("\n")
            
            f.write("###############   DISK READ (MB)     ###############\n")
            self._write_stat_mb(f, self.disk_read_stats, "Disk Read")
            f.write("\n")
            
            f.write("###############   DISK WRITE (MB)    ###############\n")
            self._write_stat_mb(f, self.disk_write_stats, "Disk Write")
            f.write("\n")
            
            # Add summary info
            f.write("=" * 60 + "\n")
            f.write(f"Total training steps: {len(self.cpu_memory_stats.history)}\n")
            if self.device.type == 'cuda' and self.gpu_available:
                f.write(f"GPU Usage Summary:\n")
                if len(self.gpu_util_stats.history) > 0:
                    avg_util = torch.tensor(self.gpu_util_stats.history).float().mean() / 100.0
                    f.write(f"  Average GPU Compute Utilization: {avg_util:.2f}%\n")
                if len(self.gpu_memory_util_stats.history) > 0:
                    avg_mem_util = torch.tensor(self.gpu_memory_util_stats.history).float().mean() / 100.0
                    f.write(f"  Average GPU Memory Utilization: {avg_mem_util:.2f}%\n")
                if len(self.gpu_power_stats.history) > 0:
                    avg_power = torch.tensor(self.gpu_power_stats.history).float().mean() / 1000.0
                    f.write(f"  Average GPU Power Consumption: {avg_power:.2f} W\n")
            if len(self.cpu_util_stats.history) > 0:
                avg_cpu_u = torch.tensor(self.cpu_util_stats.history).float().mean() / 100.0
                f.write(f"  Average CPU Utilization (process): {avg_cpu_u:.2f}%\n")
            if self.device.type == 'cuda' and self.gpu_available:
                f.write(f"\nGPU Memory Summary:\n")
                if len(self.gpu_memory_allocated_stats.history) > 0 and len(self.gpu_memory_reserved_stats.history) > 0:
                    avg_allocated = torch.tensor(self.gpu_memory_allocated_stats.history).float().mean() / (1024 * 1024)
                    avg_reserved = torch.tensor(self.gpu_memory_reserved_stats.history).float().mean() / (1024 * 1024)
                    f.write(f"  Average Allocated: {avg_allocated:.2f} MB\n")
                    f.write(f"  Average Reserved: {avg_reserved:.2f} MB\n")
                    f.write(f"  Memory Efficiency: {(avg_allocated/avg_reserved*100):.1f}%\n")
            f.write(f"\nOutput directory: {self.output_dir}\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"Resource utilization summary saved to {txt_file}")

        # Additionally, save per-step data so you can plot every step.
        self._save_per_step_csv()

    
    def _write_stat_mb(self, f, stat: utils.RunningStat, name: str) -> None:
        """Helper to write statistics in MB to file."""
        if len(stat.history) == 0:
            f.write(f"No data collected for {name}\n")
            return
        data = torch.tensor(stat.history)
        data_mb = data.to(torch.float) / (1024 * 1024)
        f.write(f"mean   : {data_mb.mean():.2f}\n")
        f.write(f"q0.25  : {data_mb.quantile(torch.tensor(0.25)):.2f}\n")
        f.write(f"q0.5   : {data_mb.quantile(torch.tensor(0.5)):.2f}\n")
        f.write(f"q0.75  : {data_mb.quantile(torch.tensor(0.75)):.2f}\n")
        f.write(f"max    : {data_mb.max():.2f}\n")
    
    def _write_stat_percentage(self, f, stat: utils.RunningStat, name: str) -> None:
        """Helper to write percentage statistics to file."""
        if len(stat.history) == 0:
            f.write(f"No data collected for {name}\n")
            return
        data = torch.tensor(stat.history)
        data_pct = data.to(torch.float) / 100.0
        f.write(f"mean   : {data_pct.mean():.2f}%\n")
        f.write(f"q0.25  : {data_pct.quantile(torch.tensor(0.25)):.2f}%\n")
        f.write(f"q0.5   : {data_pct.quantile(torch.tensor(0.5)):.2f}%\n")
        f.write(f"q0.75  : {data_pct.quantile(torch.tensor(0.75)):.2f}%\n")
        f.write(f"max    : {data_pct.max():.2f}%\n")
    
    def _log_stat_watts(self, stat: utils.RunningStat, name: str) -> None:
        """Helper to log power statistics in watts."""
        if len(stat.history) == 0:
            print(f"No data collected for {name}")
            return
        data = torch.tensor(stat.history)
        data_w = data.to(torch.float) / 1000.0  # Convert mW to W
        print(f"mean   : {data_w.mean():.2f}")
        print(f"q0.25  : {data_w.quantile(torch.tensor(0.25)):.2f}")
        print(f"q0.5   : {data_w.quantile(torch.tensor(0.5)):.2f}")
        print(f"q0.75  : {data_w.quantile(torch.tensor(0.75)):.2f}")
        print(f"max    : {data_w.max():.2f}")
    
    def _write_stat_watts(self, f, stat: utils.RunningStat, name: str) -> None:
        """Helper to write power statistics in watts to file."""
        if len(stat.history) == 0:
            f.write(f"No data collected for {name}\n")
            return
        data = torch.tensor(stat.history)
        data_w = data.to(torch.float) / 1000.0  # Convert mW to W
        f.write(f"mean   : {data_w.mean():.2f}\n")
        f.write(f"q0.25  : {data_w.quantile(torch.tensor(0.25)):.2f}\n")
        f.write(f"q0.5   : {data_w.quantile(torch.tensor(0.5)):.2f}\n")
        f.write(f"q0.75  : {data_w.quantile(torch.tensor(0.75)):.2f}\n")
        f.write(f"max    : {data_w.max():.2f}\n")

    def _save_per_step_csv(self) -> None:
        """Save per-step resource utilization time series to CSV."""
        csv_path = os.path.join(self.output_dir, "resource_util_steps.csv")

        num_steps = len(self.cpu_memory_stats.history)
        if num_steps == 0:
            logger.info("No per-step data collected; skipping CSV export.")
            return

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "gpu_util_pct",
                    "gpu_mem_util_pct",
                    "gpu_power_w",
                    "gpu_mem_alloc_mb",
                    "gpu_mem_reserved_mb",
                    "cpu_util_pct",
                    "cpu_mem_mb",
                    "disk_read_mb",
                    "disk_write_mb",
                ]
            )

            for i in range(num_steps):
                # Some stats may not be collected on CPU-only runs
                gpu_util_raw = self.gpu_util_stats.history[i] if i < len(self.gpu_util_stats.history) else 0.0
                gpu_mem_util_raw = self.gpu_memory_util_stats.history[i] if i < len(self.gpu_memory_util_stats.history) else 0.0
                gpu_power_raw = self.gpu_power_stats.history[i] if i < len(self.gpu_power_stats.history) else 0.0
                gpu_alloc_raw = self.gpu_memory_allocated_stats.history[i] if i < len(self.gpu_memory_allocated_stats.history) else 0.0
                gpu_reserved_raw = self.gpu_memory_reserved_stats.history[i] if i < len(self.gpu_memory_reserved_stats.history) else 0.0

                cpu_util_raw = self.cpu_util_stats.history[i] if i < len(self.cpu_util_stats.history) else 0.0
                cpu_mem_raw = self.cpu_memory_stats.history[i] if i < len(self.cpu_memory_stats.history) else 0.0
                disk_read_raw = self.disk_read_stats.history[i] if i < len(self.disk_read_stats.history) else 0.0
                disk_write_raw = self.disk_write_stats.history[i] if i < len(self.disk_write_stats.history) else 0.0

                writer.writerow(
                    [
                        i + 1,
                        gpu_util_raw / 100.0,
                        gpu_mem_util_raw / 100.0,
                        gpu_power_raw / 1000.0,
                        gpu_alloc_raw / (1024 * 1024),
                        gpu_reserved_raw / (1024 * 1024),
                        cpu_util_raw / 100.0,
                        cpu_mem_raw / (1024 * 1024),
                        disk_read_raw / (1024 * 1024),
                        disk_write_raw / (1024 * 1024),
                    ]
                )

        logger.info(f"Per-step resource utilization saved to {csv_path}")

    def log_loss(self, loss: torch.Tensor) -> None:
        """Log loss (optional)."""
        pass
