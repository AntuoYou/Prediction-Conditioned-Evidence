import datetime
import logging
import os
import pickle
import time
import warnings
from collections import OrderedDict, defaultdict, deque

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch import nn

from terminaltables import AsciiTable


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float32, device="npu")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('npu')

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="npu")
    size_list = [torch.tensor([0], device="npu") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="npu"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="npu")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.__dict__["meters"]:
            return self.__dict__["meters"][attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.npu.is_available():
            log_msg = self.delimiter.join([
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "mem: {memory:.0f}",
                "max mem: {max_memory:.0f}",
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.npu.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.npu.memory_allocated() / MB,
                            max_memory=torch.npu.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable)))


class HighestCheckpoint:
    def __init__(self, accelerator, model):
        self.accelerate = accelerator
        self.model = model
        self.meters = {}

    def update(self, **kwargs):
        logger = get_logger(os.path.basename(os.getcwd()) + "." + __name__)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if k not in self.meters:
                self.meters[k] = v
            if v >= self.meters[k]:
                self.meters.update({k: v})
                save_path = os.path.join(self.accelerate.project_dir, f"best_{k}.pth")
                model_state_dict = self.accelerate.get_state_dict(self.model, unwrap=True)
                self.accelerate.save(model_state_dict, save_path)
                logger.info(f"the best {k} model is saved to {save_path}")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.npu.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.npu.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def filter_mismatched_weights(model_state_dict, weight_state_dict):
    """
    过滤掉形状不匹配的权重，并返回处理后的权重字典和不匹配的键
    """
    mismatch_keys = {}
    filtered_state_dict = {}

    for key in weight_state_dict.keys():
        if key in model_state_dict:
            value_model = model_state_dict[key]
            value_state_dict = weight_state_dict[key]

            # 检查形状是否匹配
            if value_model.shape == value_state_dict.shape:
                filtered_state_dict[key] = value_state_dict
            else:
                # 记录形状不匹配的键
                mismatch_keys[key] = [value_model.shape, value_state_dict.shape]
                print(
                    f"[WARNING] Shape mismatch for {key}: model {value_model.shape} != checkpoint {value_state_dict.shape}")
        else:
            print(f"[WARNING] Key {key} not found in model")

    return filtered_state_dict, mismatch_keys


def load_checkpoint(file_name, map_location="cpu"):
    """
    加载检查点文件，支持本地路径、URL和OrderedDict
    添加对NPU设备的支持
    """
    if isinstance(file_name, str):
        if file_name.startswith("http://") or file_name.startswith("https://"):
            return torch.hub.load_state_dict_from_url(file_name, map_location=map_location)
        elif os.path.exists(file_name):
            try:
                # 对于NPU设备，确保使用正确的map_location
                if map_location == "npu" or (isinstance(map_location, str) and "npu" in map_location):
                    # 先加载到CPU，然后移动到NPU
                    checkpoint = torch.load(file_name, map_location="cpu")
                    # 如果需要，将张量移动到NPU
                    if hasattr(torch, 'npu'):
                        for k, v in checkpoint.items():
                            if isinstance(v, torch.Tensor):
                                checkpoint[k] = v.to('npu')
                    return checkpoint
                else:
                    return torch.load(file_name, map_location=map_location)
            except Exception as e:
                warnings.warn(f"Error loading checkpoint {file_name}: {e}")
                return None
        else:
            warnings.warn("Given string, only url and local path of weight are supported, skip loading.")
            return None
    elif isinstance(file_name, OrderedDict):
        return file_name

    return None


def load_state_dict(model: nn.Module, state_dict: OrderedDict, strict=False):
    """
    安全加载状态字典，处理设备不匹配和形状不匹配问题

    Args:
        model: 要加载权重的模型
        state_dict: 权重字典
        strict: 是否严格匹配，默认为False以允许部分加载
    """
    logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)

    if state_dict is None:
        logger.warn("State dict is None, skip loading")
        return

    # 获取模型当前设备
    model_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    logger.info(f"Model device: {model_device}")

    # 检查是否为NPU设备
    is_npu_device = str(model_device) == 'npu' or (hasattr(model_device, 'type') and model_device.type == 'npu')

    # 处理特殊键：不加载_classes_，因为它依赖于数据集
    if "_classes_" in state_dict:
        logger.info("Removing _classes_ from checkpoint (will use current dataset classes)")
        state_dict.pop("_classes_")

    # 将权重移动到模型所在设备
    processed_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # 根据模型设备移动权重
            if is_npu_device:
                try:
                    # 如果是NPU设备，确保权重在NPU上
                    if hasattr(torch, 'npu'):
                        processed_state_dict[key] = value.to(model_device)
                    else:
                        # 如果没有NPU支持，保持原样
                        processed_state_dict[key] = value
                except Exception as e:
                    logger.warning(f"Failed to move tensor {key} to NPU: {e}")
                    processed_state_dict[key] = value
            else:
                # 非NPU设备，正常移动
                processed_state_dict[key] = value.to(model_device)
        else:
            processed_state_dict[key] = value

    # 过滤掉形状不匹配的权重
    model_state_dict = model.state_dict()
    filtered_state_dict, mismatch_keys = filter_mismatched_weights(model_state_dict, processed_state_dict)

    # 打印加载统计信息
    total_keys = len(state_dict)
    matched_keys = len(filtered_state_dict)
    logger.info(f"Weight loading statistics: {matched_keys}/{total_keys} keys matched")

    if mismatch_keys:
        mismatch_tables = [["key name", "shape in model", "shape in checkpoint"]]

        def format_shape(shape):
            shape_str = "("
            shape_str += ", ".join([str(i) for i in shape])
            shape_str += ",)" if len(shape) == 1 else ")"
            return shape_str

        for key, (shape_model, shape_checkpoint) in mismatch_keys.items():
            mismatch_tables.append([key, format_shape(shape_model), format_shape(shape_checkpoint)])

        # 只显示前10个不匹配的键
        if len(mismatch_keys) > 10:
            mismatch_tables.append([f"... and {len(mismatch_keys) - 10} more", "", ""])

        mismatch_tables = AsciiTable(mismatch_tables)
        mismatch_tables.inner_footing_row_border = True
        logger.warning("Shape mismatch keys:\n" + mismatch_tables.table + "\n")

    # 加载过滤后的权重
    try:
        incompatible_keys = model.load_state_dict(filtered_state_dict, strict=strict)

        if strict:
            missing_keys = incompatible_keys.missing_keys
            unexpected_keys = incompatible_keys.unexpected_keys

            if len(missing_keys) == 0 and len(unexpected_keys) == 0 and len(mismatch_keys) == 0:
                logger.info("All weights loaded successfully!")
            else:
                if len(missing_keys) != 0:
                    # 只显示前10个缺失的键
                    missing_display = missing_keys[:10]
                    if len(missing_keys) > 10:
                        missing_display.append(f"... and {len(missing_keys) - 10} more")
                    logger.warning(f"Missing keys: {', '.join(missing_display)}\n")

                if len(unexpected_keys) != 0:
                    # 只显示前10个意外的键
                    unexpected_display = unexpected_keys[:10]
                    if len(unexpected_keys) > 10:
                        unexpected_display.append(f"... and {len(unexpected_keys) - 10} more")
                    logger.warning(f"Unexpected keys: {', '.join(unexpected_display)}\n")
        else:
            logger.info(f"Loaded {matched_keys} weights (non-strict mode)")

    except Exception as e:
        logger.error(f"Error loading state dict: {e}")

        # 尝试非严格模式加载（如果当前不是非严格模式）
        if strict:
            logger.info("Trying non-strict loading...")
            try:
                incompatible_keys = model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"Loaded {matched_keys} weights with non-strict mode")
            except Exception as e2:
                logger.error(f"Non-strict loading also failed: {e2}")
                raise e2
        else:
            raise e


# 添加一个新的函数用于安全加载预训练权重
def safe_load_pretrained(model, checkpoint_path, strict=False, logger=None):
    """
    安全加载预训练权重，处理各种兼容性问题
    """
    if logger is None:
        logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return model

    logger.info(f"Loading pretrained weights from: {checkpoint_path}")

    # 获取模型设备
    model_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')

    # 加载检查点
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint is None:
        logger.warning("Failed to load checkpoint. Training from scratch.")
        return model

    # 提取模型权重
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        logger.info("Found 'model' key in checkpoint")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        logger.info("Found 'state_dict' key in checkpoint")
    else:
        state_dict = checkpoint
        logger.info("Using checkpoint directly as state_dict")

    # 加载权重
    load_state_dict(model, state_dict, strict=strict)

    return model


# 修改现有的设备检查函数，添加NPU支持
def get_device():
    """获取当前可用的设备"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device('npu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# 修改SmoothedValue类中的设备处理
def get_synchronize_device():
    """获取用于同步的设备"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return "npu"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"