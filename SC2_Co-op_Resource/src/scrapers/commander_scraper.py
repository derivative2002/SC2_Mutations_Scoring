import requests
from bs4 import BeautifulSoup
import logging
import time
import sys
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
from PIL import Image, ImageDraw
import io
import random
import re
from io import BytesIO
import traceback

class CommanderSpider:
    """
    爬虫类：用于从星际争霸灰机Wiki（合作任务页面）爬取指挥官图标。
    """

    def __init__(self):
        """初始化配置"""
        self.base_url = "https://starcraft.huijiwiki.com/wiki/%E5%90%88%E4%BD%9C%E4%BB%BB%E5%8A%A1"
        self.logger = None
        self.driver = None
        self.wait = None
        self._is_driver_active = False

        # 请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://starcraft.huijiwiki.com/"
        }

        # 先设置logger
        self._setup_logger()

        # 保存目录
        self.save_dir = "指挥官"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 最后设置driver
        self._setup_driver()

    def _setup_logger(self):
        """配置日志记录器"""
        self.logger = logging.getLogger("CommanderSpider")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _setup_driver(self):
        """配置Selenium WebDriver"""
        if self.driver is not None:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
            self._is_driver_active = False

        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

            service = Service()
            self.logger.info("正在初始化Chrome WebDriver...")
            self.driver = webdriver.Chrome(service=service, options=options)
            
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                '''
            })
            
            self.wait = WebDriverWait(self.driver, 20)
            self._is_driver_active = True
            self.logger.info("Chrome WebDriver初始化成功")

        except Exception as e:
            self.logger.error(f"设置Chrome WebDriver时出错: {str(e)}")
            self._is_driver_active = False
            raise

    def __del__(self):
        """析构时关闭浏览器"""
        self.close()

    def close(self):
        """安全关闭浏览器"""
        if hasattr(self, 'driver') and self.driver and self._is_driver_active:
            try:
                self.driver.quit()
                self.logger.info("Chrome WebDriver已关闭")
            except Exception as e:
                self.logger.error(f"关闭Chrome WebDriver时出错: {str(e)}")
            finally:
                self.driver = None
                self._is_driver_active = False

    def fetch_page(self):
        """获取页面内容"""
        if not self._is_driver_active:
            self.logger.error("WebDriver未激活，尝试重新初始化...")
            self._setup_driver()
            
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self._is_driver_active:
                    raise Exception("WebDriver未激活")
                    
                self.logger.info(f"第{retry_count + 1}次尝试访问页面: {self.base_url}")
                self.driver.get(self.base_url)

                time.sleep(5)
                time.sleep(random.uniform(2, 4))

                # 模拟滚动
                for _ in range(3):
                    scroll_height = random.randint(300, 700)
                    self.driver.execute_script(f"window.scrollBy(0, {scroll_height});")
                    time.sleep(random.uniform(0.5, 1.5))

                self.driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(1)

                try:
                    self.logger.info("等待页面主体加载...")
                    element = WebDriverWait(self.driver, 30).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "mw-parser-output"))
                    )
                    self.logger.info("页面主体加载完成")
                    page_source = self.driver.page_source
                    
                    # 打印页面内容，帮助查找边框图片
                    soup = BeautifulSoup(page_source, 'html.parser')
                    content = soup.find('div', {'class': 'mw-parser-output'})
                    if content:
                        # 查找所有图片元素
                        all_images = content.find_all('img')
                        self.logger.info("页面中的所有图片URL：")
                        for img in all_images:
                            src = img.get('src', '')
                            if 'frame' in src.lower() or 'border' in src.lower():
                                self.logger.info(f"可能的边框图片: {src}")
                            elif 'commander' in src.lower():
                                self.logger.info(f"指挥官相关图片: {src}")
                    
                    if "mw-parser-output" in page_source:
                        return page_source
                    else:
                        self.logger.warning("页面源码中未找到预期的内容标记")
                        
                except Exception as e:
                    self.logger.error(f"等待页面主体超时: {e}")
                    if "no such window" in str(e):
                        self._is_driver_active = False
                        self._setup_driver()
                
                retry_count += 1
                if retry_count < max_retries:
                    self.logger.info(f"等待{5 * retry_count}秒后重试...")
                    time.sleep(5 * retry_count)
                    
            except Exception as e:
                self.logger.error(f"获取页面时发生错误: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(5 * retry_count)
                    
        self.logger.error("达到最大重试次数，获取页面失败")
        return None

    def _process_and_save_image(self, img_data, name):
        """处理并保存图片"""
        try:
            # 确保图片模式一致
            if img_data.mode != 'RGBA':
                img_data = img_data.convert('RGBA')

            # 调整图片大小为65x65（内部图标大小）
            icon_size = (65, 65)
            img_data = img_data.resize(icon_size, Image.Resampling.LANCZOS)

            # 创建一个75x75的透明背景（最终尺寸）
            final_size = (75, 75)
            final_img = Image.new('RGBA', final_size, (0, 0, 0, 0))

            # 计算居中位置
            x = (final_size[0] - icon_size[0]) // 2
            y = (final_size[1] - icon_size[1]) // 2

            # 在中心位置粘贴调整后的图标
            final_img.paste(img_data, (x, y))

            # 创建一个带有圆角蓝色边框的遮罩
            border = Image.new('RGBA', final_size, (0, 0, 0, 0))
            border_draw = ImageDraw.Draw(border)
            
            # 绘制圆角蓝色边框
            border_color = (0, 144, 255, 255)  # 蓝色，完全不透明
            border_width = 2
            radius = 10  # 圆角半径
            
            # 绘制带圆角的矩形
            border_draw.rounded_rectangle(
                [border_width//2, border_width//2, 
                 final_size[0]-1-border_width//2, final_size[1]-1-border_width//2],
                radius=radius,
                outline=border_color,
                width=border_width
            )

            # 将边框合并到最终图片上
            final_img = Image.alpha_composite(final_img, border)

            # 保存
            save_path = os.path.join(self.save_dir, f"{name}.png")
            final_img.save(save_path, 'PNG')
            self.logger.info(f"指挥官[{name}]：图片已保存")
            return True

        except Exception as e:
            self.logger.error(f"处理图片时出错: {str(e)}")
            return False

    def parse_commanders(self):
        """解析指挥官信息"""
        try:
            # 等待页面主体加载
            self.logger.info("等待页面主体加载...")
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            self.logger.info("页面主体加载完成")

            # 获取页面内容
            content = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # 查找所有指挥官图片
            self.logger.info("页面中的所有图片URL：")
            commander_images = []
            for img in content.find_all('img'):
                src = img.get('src', '')
                if 'btn-commander' in src.lower():
                    self.logger.info(f"指挥官相关图片: {src}")
                    commander_images.append(img)

            if not commander_images:
                self.logger.warning("未找到任何指挥官图片")
                return

            self.logger.info("开始解析指挥官信息...")
            commanders = []
            for img in commander_images:
                # 从图片名称中提取指挥官名称
                src = img.get('src', '')
                match = re.search(r'btn-commander-([^./]+)', src.lower())
                if not match:
                    continue
                
                commander_id = match.group(1)
                # 映射指挥官ID到中文名称
                commander_names = {
                    'raynor': '雷诺',
                    'kerrigan': '凯瑞甘',
                    'artanis': '阿塔尼斯',
                    'swann': '斯旺',
                    'zagara': '扎加拉',
                    'vorazun': '沃拉尊',
                    'karax': '凯拉克斯',
                    'abathur': '阿巴瑟',
                    'alarak': '阿拉纳克',
                    'nova': '诺娃',
                    'stukov': '斯托科夫',
                    'fenix': '菲尼克斯',
                    'dehaka': '德哈卡',
                    'horner': '霍纳与汉',
                    'tychus': '泰凯斯',
                    'zeratul': '泽拉图',
                    'stetmann': '斯台特曼',
                    'mengsk': '蒙斯克'
                }
                
                name = commander_names.get(commander_id)
                if not name:
                    continue
                
                self.logger.info(f"找到指挥官：{name}")
                commanders.append({
                    'name': name,
                    'img_url': src
                })

            self.logger.info(f"找到 {len(commanders)} 个指挥官，开始处理...")
            
            # 创建保存目录
            self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '指挥官')
            os.makedirs(self.save_dir, exist_ok=True)

            # 处理每个指挥官
            for commander in commanders:
                name = commander['name']
                img_url = self._extract_original_image_url(commander['img_url'])
                
                if not img_url:
                    continue
                    
                self.logger.info(f"处理指挥官[{name}] URL: {img_url}")
                
                try:
                    # 下载图片
                    response = requests.get(img_url)
                    response.raise_for_status()
                    
                    # 打开图片
                    img = Image.open(BytesIO(response.content))
                    
                    # 处理并保存图片
                    if not self._process_and_save_image(img, name):
                        self.logger.warning(f"指挥官[{name}]：图片处理失败")
                        continue
                    
                except Exception as e:
                    self.logger.error(f"下载图片失败：{img_url}，错误：{str(e)}")
                    self.logger.warning(f"指挥官[{name}]：图片下载失败")

        except Exception as e:
            self.logger.error(f"解析指挥官信息时出错：{str(e)}")
            traceback.print_exc()

    def _extract_original_image_url(self, thumb_url):
        # 从缩略图 URL 中提取原始图片 URL
        # 例如：从 https://huiji-thumb.huijistatic.com/starcraft/uploads/thumb/b/b5/Btn-commander-raynor.png/75px-Btn-commander-raynor.png
        # 转换为 https://huiji-public.huijistatic.com/starcraft/uploads/b/b5/Btn-commander-raynor.png
        if not thumb_url:
            return None
        
        try:
            # 替换域名
            url = thumb_url.replace('huiji-thumb', 'huiji-public')
            # 移除 /thumb/ 部分
            url = url.replace('/thumb/', '/')
            # 移除尺寸部分 (例如 /75px-)
            url = re.sub(r'/\d+px-[^/]+$', '', url)
            return url
        except Exception as e:
            self.logger.error(f"提取原始图片URL时出错：{e}")
            return None

    def _download_image(self, url):
        """下载图片"""
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        except Exception as e:
            self.logger.error(f"下载图片失败：{url}，错误：{e}")
            return None

    def run(self):
        """主流程"""
        page_html = self.fetch_page()
        if not page_html:
            self.logger.error("无法获取页面内容，结束流程。")
            return

        soup = BeautifulSoup(page_html, "html.parser")
        self.parse_commanders()
        self.logger.info("爬虫流程结束。")

if __name__ == "__main__":
    spider = CommanderSpider()
    try:
        spider.run()
    finally:
        if spider.driver:
            spider.driver.quit()
            spider.logger.info("Chrome WebDriver已关闭") 