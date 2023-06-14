import base64

import cv2
import gym
import numpy as np

from gym import spaces
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


class Dino(gym.Env):
    def __init__(self, img_size=(100, 300)):
        _chrome_options = Options()
        _chrome_options.add_argument('--mute-audio')
        _chrome_options.add_argument('disable-infobars')
        _chrome_options.add_experimental_option('detach', True)

        self._driver = webdriver.Chrome(
            executable_path=ChromeDriverManager().install(),
            options=_chrome_options
        )

        self.img_size = img_size
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.img_size[0], self.img_size[1]), dtype=np.uint8)

        # Set of actions: do nothing, jump, down
        self.action_space = spaces.Discrete(3)
        self.actions_map = [
            Keys.ARROW_RIGHT,
            Keys.ARROW_UP,
            Keys.ARROW_DOWN
        ]

        game_url = 'chrome://dino'
        try:
            self._driver.get(game_url)
        except WebDriverException:
            pass
        WebDriverWait(self._driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'runner-canvas'))
        )

    def grab_screen(self):
        getbase64Script = 'return document.querySelector("canvas.runner-canvas").toDataURL()'
        image_b64 = self._driver.execute_script(getbase64Script)
        LEADING_TEXT = 'data:image/png;base64,'
        screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64[len(LEADING_TEXT):]))))
        return screen

    def process_img(self, img):
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        gray = gray[:, :450]
        gray = cv2.resize(gray, self.img_size)
        return gray

    def get_observation(self):
        frame = self.grab_screen()
        frame = self.process_img(frame)
        return frame[np.newaxis, :, :]

    def get_score(self):
        score = 0
        score_array = self._driver.execute_script('return Runner.instance_.distanceMeter.digits')
        if len(score_array) > 0:
            score = int(''.join(score_array))
        return score

    def game_over(self):
        return self._driver.execute_script('return Runner.instance_.crashed')

    def step(self, action):
        self._driver.find_element(By.TAG_NAME, 'body').send_keys(self.actions_map[action])

        new_observation = self.get_observation()
        terminated = self.game_over()
        truncated = False
        reward = 1 if not terminated else -10

        return new_observation, reward, terminated, truncated, {'score': self.get_score()}

    def reset(self):
        self._driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.SPACE)
        return self.get_observation(), None

    def render(self):
        frame = self.grab_screen()
        frame = self.process_img(frame)

    def reload(self):
        game_url = 'chrome://dino'
        try:
            self._driver.get(game_url)
        except WebDriverException:
            pass
        WebDriverWait(self._driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'runner-canvas'))
        )