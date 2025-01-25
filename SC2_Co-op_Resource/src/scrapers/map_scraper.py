import os
import re
import logging
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import math

class MapSpider:
    def __init__(self, init_driver=False):
        self._setup_logger()
        if init_driver:
            self._setup_chrome_driver()
        else:
            self.driver = None
        self.base_url = "https://starcraft.huijiwiki.com/wiki/%E5%90%88%E4%BD%9C%E4%BB%BB%E5%8A%A1"
        
        # 创建保存目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        self.save_dir = os.path.join(project_root, "src", "resources", "images", "maps")
        os.makedirs(self.save_dir, exist_ok=True)

    def _setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger("MapSpider")
        self.logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        
        # 添加处理器到记录器
        self.logger.addHandler(console_handler)

    def _setup_chrome_driver(self):
        """设置Chrome驱动"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        self.logger.info("正在初始化Chrome WebDriver...")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.logger.info("Chrome WebDriver初始化成功")

    def _extract_original_image_url(self, thumb_url):
        """从缩略图URL中提取原始图片URL"""
        if not thumb_url:
            return None
        
        try:
            # 替换域名
            url = thumb_url.replace('huiji-thumb', 'huiji-public')
            # 移除 /thumb/ 部分
            url = url.replace('/thumb/', '/')
            # 移除尺寸部分
            url = re.sub(r'/\d+px-[^/]+$', '', url)
            return url
        except Exception as e:
            self.logger.error(f"提取原始图片URL时出错：{e}")
            return None

    def _process_and_save_image(self, img_data, name):
        """处理并保存图片，添加边框和质感效果"""
        try:
            # 确保图片模式一致
            if img_data.mode != 'RGBA':
                img_data = img_data.convert('RGBA')

            # 增加分辨率和边距
            scale = 2  # 放大2倍
            padding = 20 * scale  # 截图边距，防止裁剪到边缘
            width, height = 196 * scale, 112 * scale  # 原始尺寸的2倍
            border = 4 * scale  # 边框宽度也相应增加
            new_width = width + 2 * border
            new_height = height + 2 * border
            
            # 创建更大的画布（带边距）用于截图
            canvas_width = new_width + 2 * padding
            canvas_height = new_height + 2 * padding
            
            # 计算画布中央位置
            canvas_x = (canvas_width - new_width) // 2
            canvas_y = (canvas_height - new_height) // 2
            
            # 创建深色背景画布
            canvas_bg = Image.new('RGBA', (canvas_width, canvas_height), (8, 12, 18, 255))  # 更深的蓝黑色背景
            
            # 创建渐变背景效果
            canvas_gradient = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            canvas_draw = ImageDraw.Draw(canvas_gradient)
            
            # 为整个画布添加渐变效果
            for i in range(canvas_height):
                progress = i / canvas_height
                alpha = int(60 + progress * 120)  # 透明度范围 (60-180)
                color = (
                    int(10 + progress * 8),   # R: 10-18
                    int(15 + progress * 10),  # G: 15-25
                    int(22 + progress * 12),  # B: 22-34
                    alpha
                )
                canvas_draw.line([(0, i), (canvas_width, i)], fill=color)
            
            # 合并背景渐变
            canvas = Image.alpha_composite(canvas_bg, canvas_gradient)
            
            # 创建实际图片
            bg_color = (12, 15, 20, 255)  # 深蓝黑色背景
            new_img = Image.new('RGBA', (new_width, new_height), bg_color)
            
            # 调整原图大小，使用高质量重采样
            img_data = img_data.resize((width, height), Image.Resampling.LANCZOS)
            
            # 在新图片中央粘贴原图
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            
            # 创建临时图层用于合并
            temp = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
            temp.paste(img_data, (x, y))
            
            # 合并图层
            new_img = Image.alpha_composite(new_img, temp)
            
            # 创建绘图对象
            draw = ImageDraw.Draw(new_img)
            
            # 增强边框效果
            outer_color = (0, 200, 255, 250)  # 更亮更不透明的蓝色
            draw.rectangle([(0, 0), (new_width-1, new_height-1)], outline=outer_color, width=2*scale)
            
            # 内边框增加光晕效果
            for i in range(2):
                offset = (i + 1) * scale
                glow_color = (100, 220, 255, 160 - i * 40)
                draw.rectangle(
                    [(offset, offset), (new_width-1-offset, new_height-1-offset)],
                    outline=glow_color,
                    width=scale
                )
            
            # 增强角落装饰
            corner_size = 8 * scale
            corner_color = (180, 240, 255, 250)  # 更亮更不透明的蓝色
            corner_width = 2 * scale
            
            # 左上角
            draw.line([(0, corner_size), (0, 0), (corner_size, 0)], fill=corner_color, width=corner_width)
            # 右上角
            draw.line([(new_width-corner_size, 0), (new_width-1, 0), (new_width-1, corner_size)], fill=corner_color, width=corner_width)
            # 左下角
            draw.line([(0, new_height-corner_size), (0, new_height-1), (corner_size, new_height-1)], fill=corner_color, width=corner_width)
            # 右下角
            draw.line([(new_width-corner_size, new_height-1), (new_width-1, new_height-1), (new_width-1, new_height-corner_size)], fill=corner_color, width=corner_width)

            # 添加文字
            try:
                # 尝试加载中文字体，字体大小随分辨率放大
                font_size = int(28 * scale)  # 基础字号也随比例放大
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
            except Exception:
                try:
                    # 备用字体
                    font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", font_size)
                except Exception:
                    # 如果都失败了，使用默认字体
                    font = ImageFont.load_default()
            
            # 计算文字位置（右下角）
            text_bbox = draw.textbbox((0, 0), name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 文字位置：右下角，留出一些边距
            text_margin = 12 * scale  # 边距也随比例放大
            text_x = new_width - text_width - text_margin - border
            text_y = new_height - text_height - text_margin - border
            
            # 添加文字描边效果（完全不透明）
            outline_color = (0, 0, 0, 255)  # 黑色描边，完全不透明
            outline_width = 4 * scale  # 增加描边宽度以提高可见度
            
            # 绘制更宽的外层描边（深色）
            for offset_x in range(-outline_width-1, outline_width+2):
                for offset_y in range(-outline_width-1, outline_width+2):
                    if offset_x*offset_x + offset_y*offset_y <= (outline_width+1)*(outline_width+1):
                        draw.text((text_x + offset_x, text_y + offset_y), name, font=font, fill=(0, 0, 0, 255))
            
            # 绘制内层描边（深蓝色）
            inner_outline_color = (15, 30, 50, 255)
            for offset_x in range(-outline_width, outline_width+1):
                for offset_y in range(-outline_width, outline_width+1):
                    if offset_x*offset_x + offset_y*offset_y <= outline_width*outline_width:
                        draw.text((text_x + offset_x, text_y + offset_y), name, font=font, fill=inner_outline_color)
            
            # 添加发光效果（在描边之上）
            glow_steps = 3 * scale  # 减少发光范围
            for offset in range(1, glow_steps):
                alpha = int(160 - offset * 30)  # 提高基础不透明度
                glow_color = (150, 220, 255, alpha)
                for angle in range(0, 360, 30):  # 增加发光方向，使发光更均匀
                    rad_angle = math.radians(angle)
                    dx = int(offset * math.cos(rad_angle))
                    dy = int(offset * math.sin(rad_angle))
                    draw.text((text_x + dx, text_y + dy), name, font=font, fill=glow_color)
            
            # 绘制主文字（亮蓝色）
            text_color = (200, 240, 255, 255)  # 更亮的蓝色
            draw.text((text_x, text_y), name, font=font, fill=text_color)

            # 将处理后的图片粘贴到画布中央（使用 alpha_composite 而不是 paste）
            temp_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            temp_layer.paste(new_img, (canvas_x, canvas_y))
            canvas = Image.alpha_composite(canvas, temp_layer)

            # 保存高分辨率版本
            save_path = os.path.join(self.save_dir, f"{name}.png")
            canvas.save(save_path, 'PNG', quality=100, optimize=False)
            
            # 缩小到原始尺寸用于显示
            display_size = (width//scale + padding, height//scale + padding)
            display_img = canvas.resize(display_size, Image.Resampling.LANCZOS)
            display_path = os.path.join(self.save_dir, f"{name}_display.png")
            display_img.save(display_path, 'PNG', quality=95)
            
            self.logger.info(f"地图[{name}]：图片已保存")
            return True

        except Exception as e:
            self.logger.error(f"处理图片时出错: {str(e)}")
            return False

    def parse_maps(self):
        """解析地图信息"""
        try:
            # 获取页面内容
            content = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # 查找包含地图的表格
            table = content.find('table', class_=['wikitable', 'glow'])
            if not table:
                self.logger.warning("未找到地图表格")
                return

            # 查找所有地图按钮
            map_buttons = table.find_all('div', class_='map-button')
            self.logger.info(f"找到 {len(map_buttons)} 个地图按钮")
            
            if not map_buttons:
                self.logger.warning("未找到任何地图信息")
                return

            self.logger.info("开始解析地图信息...")
            maps = []
            for button in map_buttons:
                # 获取地图名称和链接
                title_div = button.find('div', class_='map-button-title')
                if not title_div or not title_div.find('a'):
                    continue
                    
                name = title_div.find('a').text.strip()
                
                # 获取地图预览图（从第一个a标签的img标签获取）
                a_tag = button.find('a')
                if not a_tag:
                    self.logger.warning(f"地图[{name}]：未找到链接标签")
                    continue
                    
                img = a_tag.find('img')
                if not img:
                    self.logger.warning(f"地图[{name}]：未找到图片标签")
                    continue
                    
                img_url = img.get('src', '')
                if not img_url:
                    self.logger.warning(f"地图[{name}]：未找到图片URL")
                    continue
                
                self.logger.info(f"找到地图：{name}")
                self.logger.info(f"图片URL：{img_url}")
                maps.append({
                    'name': name,
                    'img_url': img_url
                })

            self.logger.info(f"找到 {len(maps)} 个地图，开始处理...")
            
            # 处理每个地图
            for map_info in maps:
                name = map_info['name']
                img_url = map_info['img_url']
                
                if not img_url:
                    continue
                    
                self.logger.info(f"处理地图[{name}] URL: {img_url}")
                
                try:
                    # 下载图片
                    response = requests.get(img_url)
                    response.raise_for_status()
                    
                    # 打开图片
                    img = Image.open(BytesIO(response.content))
                    
                    # 处理并保存图片
                    if not self._process_and_save_image(img, name):
                        self.logger.warning(f"地图[{name}]：图片处理失败")
                        continue
                    
                except Exception as e:
                    self.logger.error(f"下载图片失败：{img_url}，错误：{str(e)}")
                    self.logger.warning(f"地图[{name}]：图片下载失败")

        except Exception as e:
            self.logger.error(f"解析地图信息时出错：{str(e)}")

    def run(self):
        """运行爬虫"""
        try:
            # 访问页面
            self.logger.info(f"访问页面: {self.base_url}")
            self.driver.get(self.base_url)
            
            # 等待页面加载完成
            self.logger.info("等待页面加载完成...")
            try:
                # 首先等待body加载
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # 然后等待表格加载
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "map-button"))
                )
                
                # 给页面一些额外的加载时间
                self.driver.implicitly_wait(5)
                
            except Exception as e:
                self.logger.error(f"页面加载超时: {str(e)}")
                return
            
            self.logger.info("页面加载完成")
            
            # 打印页面内容以供调试
            self.logger.info("解析页面内容...")
            content = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # 查找所有表格
            tables = content.find_all('table')
            self.logger.info(f"找到 {len(tables)} 个表格")
            
            # 查找包含地图的表格
            table = None
            for t in tables:
                class_list = t.get('class', [])
                self.logger.info(f"表格类名: {class_list}")
                if isinstance(class_list, list) and 'wikitable' in class_list and 'glow' in class_list:
                    table = t
                    break
            
            if not table:
                self.logger.warning("未找到地图表格")
                return
                
            # 查找所有地图按钮
            map_buttons = table.find_all('div', class_='map-button')
            self.logger.info(f"找到 {len(map_buttons)} 个地图按钮")
            
            if not map_buttons:
                self.logger.warning("未找到任何地图信息")
                return
                
            # 解析地图信息
            self.parse_maps()
            
            self.logger.info("爬虫流程结束。")
            
        except Exception as e:
            self.logger.error(f"运行爬虫时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        finally:
            if self.driver:
                self.driver.quit()
                self.logger.info("Chrome WebDriver已关闭")

    def process_local_images(self):
        """处理本地已有的背景图片"""
        try:
            # 获取背景图片目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            bg_dir = os.path.join(project_root, "src", "resources", "images", "maps_background")
            
            # 创建输出目录
            self.save_dir = os.path.join(project_root, "src", "resources", "images", "maps")
            os.makedirs(self.save_dir, exist_ok=True)
            
            # 获取所有PNG文件
            png_files = [f for f in os.listdir(bg_dir) if f.endswith('.png')]
            self.logger.info(f"找到 {len(png_files)} 个背景图片")
            
            # 处理每个图片
            for filename in png_files:
                name = os.path.splitext(filename)[0]
                self.logger.info(f"处理地图[{name}]...")
                
                try:
                    # 读取图片
                    img_path = os.path.join(bg_dir, filename)
                    img = Image.open(img_path)
                    
                    # 处理并保存图片
                    if not self._process_and_save_image(img, name):
                        self.logger.warning(f"地图[{name}]：图片处理失败")
                        continue
                        
                except Exception as e:
                    self.logger.error(f"处理图片失败：{filename}，错误：{str(e)}")
                    continue
                    
            self.logger.info("所有背景图片处理完成")
            
        except Exception as e:
            self.logger.error(f"处理本地图片时出错：{str(e)}")

if __name__ == "__main__":
    spider = MapSpider(init_driver=False)  # 不初始化 Chrome WebDriver
    try:
        # spider.run()  # 注释掉原来的网页爬取
        spider.process_local_images()  # 处理本地图片
    finally:
        if spider.driver:
            spider.driver.quit()
            spider.logger.info("Chrome WebDriver已关闭") 