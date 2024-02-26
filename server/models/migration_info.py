# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from server.models.base_model_ import Model
from server import util
import datetime as dt
import threading

class MigrationInfo(Model):

    def __init(self,):
        self.migration = 0

