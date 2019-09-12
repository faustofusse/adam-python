import webbrowser

url2 = 'http://docs.python.org/'
url = "file:///Users/mariavaleriarebora/Documents/FAUSTO/ADAM/adam-app/index.html"

# MacOS
chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
# Windows
# chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
# Linux
# chrome_path = '/usr/bin/google-chrome %s'

webbrowser.get(chrome_path).open(url)