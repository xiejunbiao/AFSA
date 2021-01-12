# -*- coding: utf-8 -*-
"""
Created on  6/20 12:58:38 2020

@author: xiejunbiao
"""
import os
import sys
pathDir = os.path.dirname(__file__)
curPath = os.path.abspath(pathDir)
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
sys.path.append(rootPath)
from server.afsa_web_server import start

    
import logging
import logging
logging.basicConfig()


def getLogger():
    logger = logging.getLogger("SENSORSDATA_IMPORTER")
    logger.propagate = False

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = getLogger()  # # initialize logging class
logger.info("starting service")

if __name__ == '__main__':
    logger.info("starting service==特征选择+人工鱼群算法+分类算法（决策树）")
    start(logger)
