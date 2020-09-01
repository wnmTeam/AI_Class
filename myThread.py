import threading
import time


class MyThread(threading.Thread):

    def __init__(self, name, processor, event=None, other_events=[], args=[]):
        super(MyThread, self).__init__(name=name)
        self.processor = processor
        self.other_events = other_events
        self.args = args

    def run(self):
        t = time.time()
        self.mydata = self.processor.processe(self.args)
        print('thread ', self.name, ': ', time.time() - t)
        for e in self.other_events:
            e.set()
