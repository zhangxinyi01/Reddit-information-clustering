from selenium import webdriver
import time
import re
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import nltk
from rake_nltk import Rake

def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    return re.sub(pattern, "", text)

# Function to extract keywords
def extract_keywords(text):
	r = Rake()
	# Extraction given the text.
	r.extract_keywords_from_text(text)
	keyword_extracted = r.get_ranked_phrases()[:5]
	# To get keyword phrases ranked highest to lowest.
	return keyword_extracted

def get_data():
	driver = webdriver.Firefox()

	# the URL of the target page to scrape
	url = 'https://www.reddit.com/r/tech/'
	# connect to the target URL in Selenium
	driver.get(url)

	#scroll down to get posts in given interval
	i = 1
	screen_height = driver.execute_script("return window.screen.height;")
	scroll_pause_time = 1 

	while True:
		driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))  
		i += 1
		time.sleep(scroll_pause_time)
		scroll_height = driver.execute_script("return document.body.scrollHeight;")
		#if (screen_height) * i > scroll_height:
		    #break
		if len(driver.find_elements(By.TAG_NAME, 'shreddit-post')) > 100:
		    break

	# retrieve the list of post HTML elements
	post_html_elements = driver.find_elements(By.TAG_NAME, 'shreddit-post')
	posts = []
	for post_html_element in post_html_elements:
		# to store the data scraped from the HTML element
		post = {}
		post['timestamp'] = post_html_element.get_attribute("created-timestamp")
		post['content'] = post_html_element.get_attribute("post-title")
		content = post_html_element.get_attribute("post-title")
		#image = post_html_element.find_element(By.CLASS_NAME, "h-full w-full object-cover")
		try:
		    image = post_html_element.find_element_by_xpath("./div/div/div/a/faceplate-img[@class='h-full w-full object-cover']")
		    image = image.get_attribute("src")
		except:
		    image = None
		post['image'] = image
		posts.append(post)
	driver.quit()
	print('Extracting keywords...')
	df = pd.DataFrame.from_dict(posts)
	# Apply the function to the 'content' column and create a new 'Keyword' column
	df['Keyword'] = df['content'].apply(extract_keywords)
	df['Keyword'] = df['Keyword'].astype(str)
	return df

