#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一个复杂的Python模块示例，用于测试agent系统的代码理解能力
"""

from __future__ import annotations
from typing import (Optional, Union, List, Dict, Any, 
                   TypeVar, Generic, Callable, Awaitable,
                   overload, runtime_checkable)
import os
import re
from datetime import datetime, timedelta
import hashlib
import hmac
import base64
import zlib
import pickle
from weakref import WeakKeyDictionary, ref

# 条件导入
try:
    import sqlalchemy as sa
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    sa = None

try:
    import redis.asyncio as redis
    from redis.exceptions import RedisError
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None

# 类型变量
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
CachedType = TypeVar('CachedType')

# 装饰器定义
def retry(max_attempts: int = 3, delay: float = 1.0, 
          exceptions: tuple = (Exception,)) -> Callable:
    """
    带指数退避的重试装饰器
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            raise last_exception
        return wrapper
    return decorator

def async_retry(max_attempts: int = 3, delay: float = 1.0,
                exceptions: tuple = (Exception,)) -> Callable:
    """
    异步版本的重试装饰器
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    await asyncio.sleep(delay * (2 ** attempt))
            raise last_exception
        return wrapper
    return decorator

def singleton(cls: type) -> type:
    """
    单例模式装饰器
    """
    instances = {}
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def cached_property(func: Callable) -> property:
    """
    缓存属性装饰器
    """
    cache_name = f'_cached_{func.__name__}'
    
    @property
    @wraps(func)
    def wrapper(self):
        if not hasattr(self, cache_name):
            setattr(self, cache_name, func(self))
        return getattr(self, cache_name)
    
    return wrapper

def validate_args(*validators):
    """
    参数验证装饰器
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i, validator in enumerate(validators):
                if i < len(args):
                    validator(args[i])
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 枚举定义
@unique
class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

@unique
class ConnectionType(str, Enum):
    TCP = "tcp"
    UDP = "udp"
    UNIX = "unix"
    WEBSOCKET = "websocket"

# 数据类
@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0
    
    def __post_init__(self):
        self._hash = hash((self.x, self.y, self.z))
    
    def distance_to(self, other: Point) -> float:
        return ((self.x - other.x) ** 2 + 
                (self.y - other.y) ** 2 + 
                (self.z - other.z) ** 2) ** 0.5

@dataclass
class Config:
    name: str
    version: str
    settings: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Config:
        path = Path(path)
        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
        elif path.suffix in ('.yml', '.yaml'):
            with open(path) as f:
                data = yaml_lib.safe_load(f)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# 抽象基类
class Serializable(ABC):
    """可序列化接口"""
    
    @abstractmethod
    def serialize(self) -> bytes:
        """序列化对象"""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Serializable:
        """反序列化对象"""
        pass
    
    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes) -> Serializable:
        """从字节创建对象"""
        pass

class Cache(ABC, Generic[K, V]):
    """缓存抽象基类"""
    
    @abstractmethod
    async def get(self, key: K) -> Optional[V]:
        pass
    
    @abstractmethod
    async def set(self, key: K, value: V, ttl: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    async def delete(self, key: K) -> bool:
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        pass

# 具体实现类
@singleton
class Logger:
    """日志管理器"""
    
    def __init__(self, name: str = "app", level: LogLevel = LogLevel.INFO):
        self.name = name
        self.level = level
        self._logger = logging.getLogger(name)
        self._handlers = []
        self._setup_logger()
    
    def _setup_logger(self):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(self.level.name)
    
    @retry(max_attempts=3)
    def log(self, level: LogLevel, message: str, **kwargs):
        method = getattr(self._logger, level.name.lower())
        method(message, extra=kwargs)
    
    @asynccontextmanager
    async def log_context(self, operation: str):
        self.log(LogLevel.INFO, f"Starting {operation}")
        try:
            yield
            self.log(LogLevel.INFO, f"Completed {operation}")
        except Exception as e:
            self.log(LogLevel.ERROR, f"Failed {operation}: {e}")
            raise

class RedisCache(Cache[str, Any]):
    """Redis缓存实现"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, prefix: str = "cache:"):
        if not HAS_REDIS:
            raise ImportError("Redis module not available")
        self._client = redis.Redis(host=host, port=port, db=db)
        self.prefix = prefix
        self._stats = {"hits": 0, "misses": 0}
    
    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    @async_retry(max_attempts=3, exceptions=(RedisError,))
    async def get(self, key: str) -> Optional[Any]:
        full_key = self._make_key(key)
        data = await self._client.get(full_key)
        if data:
            self._stats["hits"] += 1
            return pickle.loads(zlib.decompress(data))
        self._stats["misses"] += 1
        return None
    
    @async_retry(max_attempts=3, exceptions=(RedisError,))
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        full_key = self._make_key(key)
        compressed = zlib.compress(pickle.dumps(value))
        if ttl:
            await self._client.setex(full_key, ttl, compressed)
        else:
            await self._client.set(full_key, compressed)
    
    async def delete(self, key: str) -> bool:
        full_key = self._make_key(key)
        return bool(await self._client.delete(full_key))
    
    async def clear(self) -> None:
        pattern = f"{self.prefix}*"
        keys = await self._client.keys(pattern)
        if keys:
            await self._client.delete(*keys)
    
    @property
    @cached_property
    def stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return dict(self._stats)

class DatabaseSession:
    """数据库会话管理器"""
    
    def __init__(self, connection_string: str):
        if not HAS_SQLALCHEMY:
            raise ImportError("SQLAlchemy module not available")
        
        self.engine = sa.create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
        self.Session = sessionmaker(bind=self.engine)
    
    @contextmanager
    def session_scope(self):
        """提供事务范围的会话上下文"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @overload
    def query(self, model: type) -> sa.sql.Select:
        ...
    
    @overload
    def query(self, model: type, *args) -> sa.sql.Select:
        ...
    
    def query(self, model: type, *args):
        """重载的查询方法"""
        if args:
            return sa.select(*args).select_from(model)
        return sa.select(model)

class DataProcessor(Generic[T]):
    """通用数据处理类"""
    
    def __init__(self, items: Optional[List[T]] = None):
        self.items = items or []
        self._processed = WeakKeyDictionary()
    
    @validate_args(lambda x: isinstance(x, (int, float)))
    def filter_by_value(self, threshold: Union[int, float], 
                       key: Optional[Callable[[T], float]] = None) -> DataProcessor[T]:
        if key is None:
            key = lambda x: float(x) if isinstance(x, (int, float)) else 0
        
        filtered = [item for item in self.items if key(item) > threshold]
        return DataProcessor(filtered)
    
    @lru_cache(maxsize=128)
    def map_to_dict(self, key_func: Callable[[T], str]) -> Dict[str, T]:
        return {key_func(item): item for item in self.items}
    
    @singledispatch
    def process(self, data):
        """默认处理方法"""
        raise TypeError(f"Unsupported type: {type(data)}")
    
    @process.register(list)
    def _(self, data: list):
        return [self.process(item) for item in data]
    
    @process.register(dict)
    def _(self, data: dict):
        return {k: self.process(v) for k, v in data.items()}
    
    @process.register(str)
    def _(self, data: str):
        return data.upper()

# 异步上下文管理器
class AsyncConnection:
    """异步连接管理器"""
    
    def __init__(self, host: str, port: int, 
                 conn_type: ConnectionType = ConnectionType.TCP):
        self.host = host
        self.port = port
        self.conn_type = conn_type
        self._reader = None
        self._writer = None
        self._connected = False
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def connect(self):
        if self.conn_type == ConnectionType.TCP:
            self._reader, self._writer = await asyncio.open_connection(
                self.host, self.port
            )
            self._connected = True
            Logger().log(LogLevel.INFO, f"Connected to {self.host}:{self.port}")
    
    async def close(self):
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False
    
    async def send(self, data: bytes):
        if not self._connected:
            raise ConnectionError("Not connected")
        self._writer.write(data)
        await self._writer.drain()
    
    async def receive(self, n: int = 1024) -> bytes:
        if not self._connected:
            raise ConnectionError("Not connected")
        return await self._reader.read(n)

# 混入类
class JSONMixin:
    """JSON序列化混入"""
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)
    
    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(**data)

class HashMixin:
    """哈希计算混入"""
    
    def compute_hash(self, algorithm: str = 'sha256') -> str:
        hasher = hashlib.new(algorithm)
        hasher.update(str(self.__dict__).encode())
        return hasher.hexdigest()
    
    def verify_hash(self, hash_value: str, algorithm: str = 'sha256') -> bool:
        return hmac.compare_digest(self.compute_hash(algorithm), hash_value)

# 最终的综合类
class SecureDocument(JSONMixin, HashMixin, Serializable):
    """安全文档类"""
    
    def __init__(self, doc_id: str, content: str, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.version = 1
        self._signature = None
    
    def update_content(self, new_content: str):
        self.content = new_content
        self.updated_at = datetime.now()
        self.version += 1
        self._signature = None
    
    @retry(max_attempts=3, delay=0.5)
    def sign(self, secret_key: bytes) -> str:
        if not self._signature:
            message = f"{self.doc_id}:{self.version}:{self.updated_at.timestamp()}"
            self._signature = hmac.new(
                secret_key,
                message.encode(),
                hashlib.sha256
            ).hexdigest()
        return self._signature
    
    def serialize(self) -> bytes:
        data = {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            '_signature': self._signature
        }
        return zlib.compress(pickle.dumps(data))
    
    def deserialize(self, data: bytes) -> SecureDocument:
        decompressed = zlib.decompress(data)
        data_dict = pickle.loads(decompressed)
        
        doc = SecureDocument(
            doc_id=data_dict['doc_id'],
            content=data_dict['content'],
            metadata=data_dict['metadata']
        )
        doc.created_at = datetime.fromisoformat(data_dict['created_at'])
        doc.updated_at = datetime.fromisoformat(data_dict['updated_at'])
        doc.version = data_dict['version']
        doc._signature = data_dict['_signature']
        return doc
    
    @classmethod
    def from_bytes(cls, data: bytes) -> SecureDocument:
        doc = cls("", "")
        return doc.deserialize(data)

# 函数重载示例
@overload
def process_input(data: str) -> str:
    ...

@overload
def process_input(data: bytes) -> bytes:
    ...

@overload
def process_input(data: Dict[str, Any]) -> Dict[str, Any]:
    ...

def process_input(data):
    if isinstance(data, str):
        return data.strip().lower()
    elif isinstance(data, bytes):
        return base64.b64encode(data)
    elif isinstance(data, dict):
        return {k.lower(): v for k, v in data.items()}
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

# 主程序入口
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # 测试日志
        logger = Logger()
        async with logger.log_context("testing"):
            logger.log(LogLevel.INFO, "This is a test")
        
        # 测试数据处理器
        processor = DataProcessor([1, 2, 3, 4, 5])
        filtered = processor.filter_by_value(3)
        print(f"Filtered: {filtered.items}")
        
        # 测试文档
        doc = SecureDocument("doc1", "Hello, World!", {"author": "test"})
        doc.sign(b"secret")
        
        # 测试序列化
        serialized = doc.serialize()
        restored = SecureDocument.from_bytes(serialized)
        print(f"Restored doc: {restored.doc_id}, content: {restored.content}")
        
        # 测试异步连接
        try:
            async with AsyncConnection("localhost", 8888) as conn:
                await conn.send(b"Hello")
        except ConnectionError:
            print("Connection failed (expected if server not running)")
    
    asyncio.run(main())
