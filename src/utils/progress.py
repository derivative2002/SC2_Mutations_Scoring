from tqdm import tqdm
from typing import Optional, Iterator, Any

class ProgressBar:
    """进度条包装器"""
    def __init__(
        self,
        iterable: Optional[Iterator] = None,
        total: Optional[int] = None,
        desc: str = '',
        leave: bool = True
    ):
        self.pbar = tqdm(
            iterable,
            total=total,
            desc=desc,
            leave=leave,
            ncols=100,
            ascii=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
    
    def update(self, n: int = 1):
        """更新进度"""
        self.pbar.update(n)
    
    def set_description(self, desc: str):
        """设置描述"""
        self.pbar.set_description(desc)
    
    def set_postfix(self, **kwargs):
        """设置后缀信息"""
        self.pbar.set_postfix(**kwargs)
    
    def close(self):
        """关闭进度条"""
        self.pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class EpochProgressBar(ProgressBar):
    """训练轮次进度条"""
    def __init__(self, num_epochs: int, desc: str = 'Training'):
        super().__init__(total=num_epochs, desc=desc)
        self.num_epochs = num_epochs
    
    def update_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        best_val_acc: float
    ):
        """更新训练指标"""
        self.set_description(f'Epoch [{epoch}/{self.num_epochs}]')
        self.set_postfix(
            train_loss=f'{train_loss:.4f}',
            val_loss=f'{val_loss:.4f}',
            val_acc=f'{val_acc:.4f}',
            best=f'{best_val_acc:.4f}'
        )
        self.update()

class BatchProgressBar(ProgressBar):
    """批次进度条"""
    def __init__(self, num_batches: int, desc: str = 'Training'):
        super().__init__(total=num_batches, desc=desc, leave=False)
    
    def update_loss(self, loss: float):
        """更新损失值"""
        self.set_postfix(loss=f'{loss:.4f}') 