# Marine Image Retrieval API

一个基于 FastAPI 的海洋图像识别服务，支持：

- 基于 BioCLIP + FAISS 的检索
- sonar / biological 路由分类
- sonar 图像分类
- biological 图像的 fish / coral 二分类
- fish / coral 对应 detector 推理
- 可选的 BioCLIP2 生物术语补充
- 多模块结果融合输出
## 项目结构
```text
app/
  api/          # FastAPI 路由
  core/         # 配置与全局状态
  services/     # 模型加载、检索、分类、融合等核心逻辑
  main.py       # 应用入口
```
`app/`是 Python 主包目录。

*   `app/api/`部署 HTTP 接口，也就是 FastAPI 路由。

*   `app/api/routes.py`用于定义 /health，定义 /predict，校验上传文件是否为图片，把图片存到临时目录，调用 run_full_pipeline(...)，将结果包装成 JSON 返回


`app/core/`放全局基础设施。

*   `app/core/config.py`为配置中心，存储各种配置参数和路径。

*   `app/core/state.py`是运行时状态缓存。存储BioCLIP retrieval 模型，FAISS index，metadata，router model，sonar model，fish/coral classifier，fish detector，coral detector，BioCLIP2 model，预计算 text features，当前运行 device。
让所有 service 文件共享模型和索引，而不需要到处传几十个参数。

`app/services/`是核心业务逻辑层。

*   `app/services/retrieval.py`负责 FAISS + BioCLIP 检索链。主要完成：加载 BioCLIP retrieval 模型， FAISS index，metadata，对输入图片编码，检索 topk，构造 retrieval module，断是否 db_hit。**数据库检索**，命中则直接返回

*   `app/services/router.py`负责 router 分类：sonar / biological，对图片做 router 推理，根据阈值决定调用sonar还是biological识别。



*   `app/services/bioclip2_service.py`负责 BioCLIP2 相关逻辑。从 shard 里提取术语，给出生物术语匹配结果。

*   `app/services/pipeline.py`这是总调度中心，执行整个完整的检测流程，最后对结果做fusion并组装 final_result。


`app/main.py`是程序入口。创建 FastAPI(...)，include_router(...)，在 startup 时加载所有模型和索引，启动 uvicorn。


## 环境准备
建议使用 conda：

```bash
conda env create -f environment.yml
conda activate marine-api
```

## 配置环境变量

复制示例文件并修改：

`cp .env.example .env`

然后按你的实际路径填写模型、索引和数据文件路径。

你也可以直接 export 环境变量，例如：

```bash
export YOLOV5_DIR=/path/to/yolov5
export MODEL_DIR=/path/to/models/bioclip
```
## 启动服务
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
## API
#### 健康检查
```bash
GET /health
```

#### 预测接口
```bash
POST /predict
```

表单字段：

* `file`: 图片文件

## 说明

本仓库默认不包含大模型权重、FAISS 索引和数据衍生文件。
请根据项目说明自行下载并配置路径。
## 模型下载
