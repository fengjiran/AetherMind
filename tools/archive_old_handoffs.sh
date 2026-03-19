#!/bin/bash
#
# Handoff 归档脚本
# 用途：将超过30天的 closed/superseded handoff 移动到 archived/ 子目录
# 用法：./archive_old_handoffs.sh [--dry-run] [--days=N]
#

set -euo pipefail

# 配置
HANDOFF_DIR="docs/agent/handoff/workstreams"
ARCHIVE_DIR="archived"
DEFAULT_DAYS=30
DRY_RUN=false

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 解析参数
DAYS=$DEFAULT_DAYS
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --days=*)
            DAYS="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --dry-run       预览模式，不实际移动文件"
            echo "  --days=N        设置归档阈值天数（默认: $DEFAULT_DAYS）"
            echo "  -h, --help      显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查 handoff 目录是否存在
if [[ ! -d "$HANDOFF_DIR" ]]; then
    echo -e "${RED}错误: Handoff 目录不存在: $HANDOFF_DIR${NC}"
    exit 1
fi

# 计算阈值日期（ISO 8601 格式）
THRESHOLD_DATE=$(date -d "$DAYS days ago" +%Y-%m-%d 2>/dev/null || date -v-${DAYS}d +%Y-%m-%d)

echo "======================================"
echo "Handoff 归档脚本"
echo "======================================"
echo "阈值日期: $THRESHOLD_DATE (超过 $DAYS 天)"
echo "归档目录: $ARCHIVE_DIR/"
echo "模式: $([ "$DRY_RUN" = true ] && echo "预览 (dry-run)" || echo "执行")"
echo ""

# 统计变量
TOTAL_FILES=0
ARCHIVED_FILES=0
SKIPPED_FILES=0
ERROR_FILES=0

# 创建归档目录
if [[ "$DRY_RUN" = false ]]; then
    mkdir -p "$HANDOFF_DIR/$ARCHIVE_DIR"
fi

# 查找所有 handoff 文件
find "$HANDOFF_DIR" -name "*.md" -type f | while read -r file; do
    # 跳过 archived 目录中的文件
    if [[ "$file" == *"/$ARCHIVE_DIR/"* ]]; then
        continue
    fi
    
    ((TOTAL_FILES++)) || true
    
    # 提取 frontmatter 中的状态和日期
    STATUS=$(grep -A 20 "^---$" "$file" | grep "^status:" | head -1 | awk '{print $2}' || echo "")
    CREATED_AT=$(grep -A 20 "^---$" "$file" | grep "^created_at:" | head -1 | awk '{print $2}' || echo "")
    CLOSED_AT=$(grep -A 20 "^---$" "$file" | grep "^closed_at:" | head -1 | awk '{print $2}' || echo "")
    
    # 只处理 closed 或 superseded 状态的文件
    if [[ "$STATUS" != "closed" && "$STATUS" != "superseded" ]]; then
        ((SKIPPED_FILES++)) || true
        continue
    fi
    
    # 确定参考日期（优先使用 closed_at，否则使用 created_at）
    if [[ -n "$CLOSED_AT" && "$CLOSED_AT" != "null" ]]; then
        REF_DATE=$(echo "$CLOSED_AT" | cut -d'T' -f1)
    elif [[ -n "$CREATED_AT" && "$CREATED_AT" != "null" ]]; then
        REF_DATE=$(echo "$CREATED_AT" | cut -d'T' -f1)
    else
        echo -e "${YELLOW}警告: 无法解析日期 - $file${NC}"
        ((ERROR_FILES++)) || true
        continue
    fi
    
    # 比较日期
    if [[ "$REF_DATE" < "$THRESHOLD_DATE" ]]; then
        # 获取相对路径
        REL_PATH="${file#$HANDOFF_DIR/}"
        ARCHIVE_PATH="$HANDOFF_DIR/$ARCHIVE_DIR/$REL_PATH"
        
        echo -e "${GREEN}[归档] $file${NC}"
        echo "  状态: $STATUS"
        echo "  参考日期: $REF_DATE"
        
        if [[ "$DRY_RUN" = false ]]; then
            # 创建目标目录
            mkdir -p "$(dirname "$ARCHIVE_PATH")"
            
            # 移动文件
            if mv "$file" "$ARCHIVE_PATH"; then
                ((ARCHIVED_FILES++)) || true
                echo "  → 已移动到: $ARCHIVE_PATH"
            else
                echo -e "${RED}  ✗ 移动失败${NC}"
                ((ERROR_FILES++)) || true
            fi
        else
            ((ARCHIVED_FILES++)) || true
            echo "  → (预览) 将移动到: $ARCHIVE_PATH"
        fi
    else
        ((SKIPPED_FILES++)) || true
    fi
done

echo ""
echo "======================================"
echo "归档报告"
echo "======================================"
echo "扫描文件总数: $TOTAL_FILES"
echo -e "${GREEN}归档文件数: $ARCHIVED_FILES${NC}"
echo "跳过文件数: $SKIPPED_FILES"
echo -e "${RED}错误文件数: $ERROR_FILES${NC}"

if [[ "$DRY_RUN" = true ]]; then
    echo ""
    echo -e "${YELLOW}注意: 这是预览模式，文件未被实际移动。${NC}"
    echo "要执行归档，请移除 --dry-run 选项。"
fi

echo ""

# 退出码
if [[ $ERROR_FILES -gt 0 ]]; then
    exit 1
fi