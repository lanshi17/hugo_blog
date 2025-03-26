#!/usr/bin/env bash

PDF_DIR="./static/pdfs"
DATA_DIR="./data/pdfinfo"
mkdir -p "$DATA_DIR"

for pdf in "$PDF_DIR"/*.pdf; do
  filename=$(basename "$pdf" .pdf)
  info=$(pdfinfo "$pdf")

  # 提取页数
  pages=$(echo "$info" | grep -i 'Pages' | awk '{print $2}')

  # 提取页面尺寸
  page_width=$(echo "$info" | grep -i 'Page.*size:' | awk '{print $3}')
  page_height=$(echo "$info" | grep -i 'Page.*size:' | awk '{print $5}')

  # 移除非数字字符并保留浮点数
  page_width=$(echo "$page_width" | grep -oE '[0-9.]+')
  page_height=$(echo "$page_height" | grep -oE '[0-9.]+')

  # 设置默认值
  pages=${pages:-1}
  page_width=${page_width:-612.00}
  page_height=${page_height:-792.00}

  # 生成YAML数据文件
  cat > "$DATA_DIR/$filename.yaml" << EOF
pdfinfo:
  pages: $pages
  width: $page_width
  height: $page_height
EOF

  echo "生成: $DATA_DIR/$filename.yaml (尺寸: ${page_width}x${page_height} pts)"
done