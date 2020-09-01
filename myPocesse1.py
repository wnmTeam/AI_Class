import multiprocessing
import time

class MyProcesse(multiprocessing.Process):

    def __init__(self, name, q_in, q_out):
        super(MyProcesse, self).__init__(name=name)
        self.q_in = q_in
        self.q_out = q_out

    def run(self):
        self.processer = self.q_in.get()
        while True:
            if not self.q_in.empty:
                t = time.time()
                self.mydata = self.processer.processe(self.q_in.get())
                print('processe ', self.name, ': ', time.time() - t)


