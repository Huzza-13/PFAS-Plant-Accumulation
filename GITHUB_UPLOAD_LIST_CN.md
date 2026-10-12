# GitHub 上传清单：18 个脚本，重新编号 01-18

目标仓库：https://github.com/Huzza-13/PFAS-Plant-Accumulation

按作者要求，不纳入最后三个研究脚本（水稻/蔬菜、小麦、世界地图），其他配套文件保留。原始目录和先前整理包没有修改。新编号按实验室分析、田间分析及两者比较排列，不再混用两位与三位序号。

**网页替换记录（2026-10-12）：**旧的 16 个脚本先删除（提交 `36b5660`），再上传新版 18 个研究脚本和两个辅助文件（提交 `391a0e9`）。配套文件随后同步，以仓库实际提交为准；此前网络失败的记录是历史状态。

## 新旧编号与完整文件名

| 新编号 | 新文件名（位于 scripts/） | 原编号 | 内容 |
|---|---|---|---|
| 01 | 01_Molecular_correlation.py | 03 | 描述符相关性与类别 |
| 02 | 02_Laboratory_model_screening.py | 11 | 实验室算法筛选 |
| 03 | 03_Feature_block_models.py | 02 | 预测变量分组模型 |
| 04 | 04_Predictor_domain_sensitivity.py | 13 | 分组比较图 |
| 05 | 05_Laboratory_nested_XGBoost.py | 05 | 实验室嵌套验证与调参 |
| 06 | 06_Laboratory_OOF_performance.py | 06 | 实验室折外预测图 |
| 07 | 07_Laboratory_fold_diagnostics.py | 14 | 实验室验证折诊断 |
| 08 | 08_Laboratory_final_XGBoost.py | 07 | 实验室最终解释模型 |
| 09 | 09_Laboratory_overall_SHAP.py | 08 | 实验室整体 SHAP |
| 10 | 10_Laboratory_combined_SHAP.py | 008 | 实验室组合 SHAP 图 |
| 11 | 11_Molecular_SHAP_chain_class.py | 17 | 碳链/类别分子 SHAP |
| 12 | 12_Global_model_tissue_SHAP.py | 19 | 全局模型组织分层 SHAP |
| 13 | 13_Field_nested_modeling.py | 001 | 田间四算法嵌套验证，普通 GB |
| 14 | 14_Field_OOF_predictions.py | 002 | 田间折外预测图 |
| 15 | 15_Field_SHAP.py | 003 | 田间 SHAP：参数来源待确认 |
| 16 | 16_Field_SHAP_beeswarm.py | 005 | 田间蜂群图：参数来源待确认 |
| 17 | 17_Laboratory_field_comparison.py | 004 | 实验室/田间比较：精确田间表待生成 |
| 18 | 18_Field_validation_SI.py | 006 | 田间补充验证图 |

完整原文件名见 `docs/script_numbering.csv`。重新编号没有重新训练或修改研究脚本内容，原有图像输出文件名保留。运行入口和基础核验同步更新。

## 上传范围

把本目录的实际文件作为仓库根目录内容上传，不要只上传 ZIP：

```text
README.md
GITHUB_UPLOAD_LIST_CN.md
requirements.txt
run.py
check_package.py
.gitignore
RIGHTS.md
scripts/              18 个研究脚本 + 2 个必需辅助文件
data/                 三份未修改工作簿、原地图组件与说明
reference_results/    已有保存结果及快照说明
docs/                 来源、编号、环境与检查记录
```

两个辅助文件 `project_paths.py`、`pfas_chain_lengths.py` 不能遗漏。膳食/地图数据及旧表格保留，但没有对应三个分析脚本，不能标为本包已复现结果。

不上传 `results/`、虚拟环境、缓存、第三方全文论文、历史/探索性代码或外部整理工具。

## 替换策略

仅替换指定仓库当前默认分支文件，不删除仓库、Issues、提交历史或其他分支。先读取并保存旧提交，完整备份原仓库；再以正常后继提交移除旧文件并加入本包。不要强制推送或清除历史。删除和新增合并为一个提交可保证最终目录整体替换，同时避免仓库中途停留在空目录。

原默认分支为 main，替换前提交为 `78bd8c78269aa8974887e327edf20262cc32d71d`。已在本地保存并核验完整 Git 历史备份，旧内容也可从该提交恢复。本次不删除或重写提交历史。

## 验证边界

- 普通 GB 参数检查通过，但尚未重训核对历史指标；最终最佳田间模型仍是 XGBoost。
- 15/16 参数来源、17 精确输入表待确认；没有绕过审核检查。
- 保存模型有版本兼容性警告。基础核验和历史回放不等于干净环境端到端复现。
- 原始数据来源、许可和机构公开要求需作者确认，未代作者选择许可证。
