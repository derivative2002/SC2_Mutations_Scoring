# -*- coding: utf-8 -*-
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
from PIL import Image
import io
import random

class MutationSpider:
    """
    爬虫类：用于从星际争霸灰机Wiki（突变因子页面）爬取并组合"主图 + 边框"。
    注意：不使用硬编码映射，而是直接从HTML中的<a>标签获取中文名称。
    
    核心流程：
      1) 用Selenium打开页面并等待加载，获取HTML
      2) 用BeautifulSoup解析HTML，找到带有 position:relative 的div，提取主图/边框图URL及中文名称
      3) 将下载的主图 + 边框图合成为一张图片，并以"中文名称.png"命名保存
    """

    def __init__(self):
        """初始化各项配置"""
        self.base_url = "https://starcraft.huijiwiki.com/wiki/%E5%90%88%E4%BD%9C%E4%BB%BB%E5%8A%A1/%E7%AA%81%E5%8F%98%E5%9B%A0%E5%AD%90"
        self.logger = None
        self.driver = None
        self.wait = None
        self._is_driver_active = False  # 添加标志来追踪driver状态

        # 用于requests请求的headers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://starcraft.huijiwiki.com/"
        }

        # 下载的合成图片保存目录
        self.save_dir = "突变因子"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 日志
        self._setup_logger()
        # 配置Selenium
        self._setup_driver()

    def _setup_logger(self):
        """配置日志记录器"""
        self.logger = logging.getLogger("MutationSpider")
        self.logger.setLevel(logging.INFO)
        # 清理已有的handler，防止重复添加
        self.logger.handlers = []

        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _setup_driver(self):
        """配置Selenium WebDriver（Chrome）"""
        if self.driver is not None:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
            self._is_driver_active = False

        try:
            options = Options()
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-software-rasterizer")
            options.add_argument("--ignore-certificate-errors")
            options.add_argument("--allow-running-insecure-content")
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

                # 增加页面加载等待时间
                time.sleep(5)

                # 添加随机延迟，模拟人类行为
                time.sleep(random.uniform(2, 4))

                # 模拟真实的浏览行为
                for _ in range(3):  # 随机滚动3次
                    scroll_height = random.randint(300, 700)
                    self.driver.execute_script(f"window.scrollBy(0, {scroll_height});")
                    time.sleep(random.uniform(0.5, 1.5))

                # 最后滚动到顶部
                self.driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(1)

                # 显式等待页面主体出现
                try:
                    self.logger.info("等待页面主体加载...")
                    element = WebDriverWait(self.driver, 30).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "mw-parser-output"))
                    )
                    self.logger.info("页面主体加载完成")
                    page_source = self.driver.page_source
                    
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
                if "no such window" in str(e):
                    self._is_driver_active = False
                    self._setup_driver()
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(5 * retry_count)
                    
        self.logger.error("达到最大重试次数，获取页面失败")
        return None

    def parse_mutations(self, soup):
        """解析突变因子信息"""
        if not soup:
            self.logger.error("页面内容为空，无法解析。")
            return

        self.logger.info("开始解析突变因子信息...")

        # 找到内容所在的主容器
        content = soup.find('div', {'class': 'mw-parser-output'})
        if not content:
            self.logger.error("未找到内容区域(mw-parser-output)，解析中断。")
            return

        # 找到所有突变因子容器
        mutation_divs = []
        
        # 找到所有包含突变因子的flex容器
        flex_containers = content.find_all('div', style=lambda x: x and 'display:flex' in x)
        
        # 打印第一个容器的结构以便调试
        if flex_containers:
            self.logger.info(f"第一个flex容器的HTML结构：\n{flex_containers[0].prettify()}")
        
        for container in flex_containers:
            # 在每个flex容器中查找图标和名称
            icon_container = container.find('div', style=lambda x: x and 'position:relative' in x)
            # 直接获取flex容器下的a标签
            a_tag = container.find('a')
            
            if icon_container and a_tag:
                # 确保这个容器包含了我们需要的元素
                if (icon_container.find('div', style=lambda x: x and 'border-radius:10px' in x) and 
                    icon_container.find('div', {'class': 'icon-mask'})):
                    mutation_divs.append((icon_container, a_tag))

        if not mutation_divs:
            self.logger.warning("未找到任何突变因子容器。")
            return

        self.logger.info(f"找到 {len(mutation_divs)} 个突变因子容器，开始处理...")

        for idx, (icon_container, a_tag) in enumerate(mutation_divs, 1):
            try:
                # 获取中文名称
                chinese_name = a_tag.text.strip()
                if not chinese_name:
                    # 如果文本为空，尝试获取title属性
                    chinese_name = a_tag.get('title', '').strip()
                    if chinese_name.startswith('#'):
                        chinese_name = chinese_name[1:]  # 移除开头的#号
                
                if not chinese_name:
                    self.logger.warning(f"第{idx}个容器：无法找到有效的名称")
                    # 打印当前容器的结构以便调试
                    self.logger.info(f"当前容器的HTML结构：\n{icon_container.parent.prettify()}")
                    continue

                self.logger.info(f"处理突变因子：{chinese_name}")

                # 获取图标和边框
                icon_div = icon_container.find('div', style=lambda x: x and 'border-radius:10px' in x)
                if not icon_div:
                    self.logger.warning(f"突变因子[{chinese_name}]：未找到icon_div")
                    continue
                    
                icon_img = icon_div.find('img')
                if not icon_img:
                    self.logger.warning(f"突变因子[{chinese_name}]：未找到icon_img标签")
                    continue

                mask_div = icon_container.find('div', {'class': 'icon-mask'})
                if not mask_div:
                    self.logger.warning(f"突变因子[{chinese_name}]：未找到mask_div")
                    continue
                    
                mask_img = mask_div.find('img')
                if not mask_img:
                    self.logger.warning(f"突变因子[{chinese_name}]：未找到mask_img标签")
                    continue

                # 获取图片URL
                icon_url = self._extract_original_image_url(icon_img)
                mask_url = self._extract_original_image_url(mask_img)

                self.logger.info(f"突变因子[{chinese_name}] URL:\n  图标：{icon_url}\n  边框：{mask_url}")

                if not icon_url or not mask_url:
                    self.logger.warning(f"突变因子[{chinese_name}]：无法获取图片URL")
                    continue

                # 下载并处理图片
                icon_pic = self._download_image(icon_url)
                mask_pic = self._download_image(mask_url)

                if not icon_pic or not mask_pic:
                    self.logger.warning(f"突变因子[{chinese_name}]：图片下载失败")
                    continue

                # 处理图片
                self._process_and_save_image(icon_pic, mask_pic, chinese_name, idx)

            except Exception as e:
                self.logger.error(f"处理第{idx}个突变因子时出错: {str(e)}")
                continue

    def _process_and_save_image(self, icon_img, mask_img, name, idx):
        """处理并保存图片"""
        try:
            # 确保图片模式一致
            if icon_img.mode != 'RGBA':
                icon_img = icon_img.convert('RGBA')
            if mask_img.mode != 'RGBA':
                mask_img = mask_img.convert('RGBA')

            # 设置最终图片大小
            final_size = (71, 73)  # 边框原始大小

            # 调整图片大小
            icon_img = icon_img.resize((55, 55), Image.Resampling.LANCZOS)  # 减小图标尺寸
            mask_img = mask_img.resize(final_size, Image.Resampling.LANCZOS)

            # 创建新图片
            result = Image.new('RGBA', final_size, (0, 0, 0, 0))

            # 计算图标位置（居中）
            icon_x = (final_size[0] - 55) // 2
            icon_y = (final_size[1] - 55) // 2

            # 先贴图标
            result.paste(icon_img, (icon_x, icon_y), icon_img)
            
            # 再贴边框
            result.paste(mask_img, (0, 0), mask_img)

            # 保存
            save_path = os.path.join(self.save_dir, f"{name}.png")
            result.save(save_path, 'PNG')
            self.logger.info(f"完成第{idx}个突变因子[{name}]的合成")

        except Exception as e:
            self.logger.error(f"处理图片时出错: {str(e)}")
            raise

    def _extract_original_image_url(self, img_tag):
        """
        从<img>标签中提取完整图片地址（去除thumb尺寸并改成原图），
        优先取srcset中的第一段，如果不存在srcset则使用src。
        """
        image_url = ""
        if img_tag.get('srcset'):
            # 例如 "https://huiji-public.huijistatic.com/starcraft/uploads/8/85/Househunters_coop.png 1.5x"
            image_url = img_tag['srcset'].split()[0]
        else:
            image_url = img_tag.get('src', '')

        if not image_url:
            return None

        # 如果是 // 开头，补上 https:
        if image_url.startswith('//'):
            image_url = 'https:' + image_url

        # 如果带 /thumb/，则去除 "/thumb/..." 并把结尾改成 .png 等
        if '/thumb/' in image_url:
            parts = image_url.split('/')
            if 'thumb' in parts:
                thumb_index = parts.index('thumb')
                # 确保后面有足够的元素
                if len(parts) >= thumb_index + 3:
                    # 去掉 'thumb' 和 后续尺寸路径，用 ".png" 结尾（也可能是jpg，根据实际需要修改）
                    image_url = '/'.join(parts[:thumb_index] + parts[thumb_index+1:-2]) + '.png'

        return image_url

    def _download_image(self, url):
        """
        用 requests 下载图片，返回 PIL.Image 对象，若失败返回 None
        """
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        except Exception as e:
            self.logger.error(f"下载图片失败：{url}，错误：{e}")
            return None

    def run(self):
        """
        主流程：获取页面 → 一次性解析突变因子 → 下载并合成主图 + mask → 以中文名保存
        """
        page_html = self.fetch_page()
        if not page_html:
            self.logger.error("无法获取页面内容，结束流程。")
            return

        soup = BeautifulSoup(page_html, "html.parser")
        self.parse_mutations(soup)
        self.logger.info("爬虫流程结束。")

if __name__ == "__main__":
    spider = MutationSpider()
    spider.run()
    # 若需要手动关闭浏览器，可调 spider.close() 或等待 __del__ 调用