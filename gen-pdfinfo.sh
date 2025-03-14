#!/usr/bin/env bash

PDF_DIR="./static/pdfs"
DATA_DIR="./data/pdfinfo"
mkdir -p "$DATA_DIR"

for pdf in "$PDF_DIR"/*.pdf; do
  filename=$(basename "$pdf" .pdf)
  info=$(pdfinfo "$pdf")

  # 提取页数
  pages=$(echo "$info" | grep -i 'Pages' | awk '{print $2}')

  # 提取页面尺寸（兼容小数和单位）
  page_size_line=$(pdfinfo "$pdf" | grep -i 'Page.*size:' | head -n1)
  
  # 使用 awk 精确提取宽高（第三列和第五列）
  page_width=$(echo "$page_size_line" | awk '{print $3}')
  page_height=$(echo "$page_size_line" | awk '{print $5}')

  # 移除可能的非数字字符（如 pts）
  page_width=$(echo "$page_width" | grep -oE '[0-9.]+' | head -n1)
  page_height=$(echo "$page_height" | grep -oE '[0-9.]+' | head -n1)

  # 设置默认值（US Letter尺寸，单位：点）
  page_width=${page_width:-612.00}
  page_height=${page_height:-792.00}

  # 生成 YAML 数据文件（保留浮点数精度）
  cat << EOF > "$DATA_DIR/$filename.yaml"
    pdfinfo:
        pages: ${pages:-1}
        width: $page_width
        height: $page_height
EOF

  echo "Generated: $DATA_DIR/$filename.yaml (Size: ${page_width}x${page_height} pts)"
done