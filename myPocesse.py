import multiprocessing
import time

class MyProcesse(multiprocessing.Process):

    def __init__(self, name, processe, event=None, other_events=[], args=[]):
        super(MyProcesse, self).__init__(name=name)
        self.processe = processe
        self.event = event
        self.mydata = []
        self.other_events = other_events
        self.args = args

    def run(self):
        if self.event != None:
            # print('thread ', self.name, ' waiting...')
            self.event.wait()
        # print('thread ', self.name, ' running...')
        t = time.time()
        self.mydata = self.processe(self.args)
        # print('thread ', self.name, ' finish.')
        print('processe ', self.name, ': ', time.time() - t)
        for e in self.other_events:
            e.set()

