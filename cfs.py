from time import time
import bintrees
import json
import getopt
import sys

tasks = []
time_slice = 45
num_timeslices = 8
tree = bintrees.RBTree()
exit = False


class ProcessTree:
    def __init__(self, tasks=None):
        self.tree = bintrees.RBTree()
        self.stats = tasks.get('stats')
        if tasks:
            self.add_tasks(tasks['tasks'])

    def task_weight(self, task, tasks):
        return task.weight / sum(task.weight for task in tasks)

    def add_tasks(self, task_dicts):
        if type(task_dicts) == dict:
            tasks = [Task(**task_dicts)]
        else:
            tasks = [Task(**task) for task in task_dicts]

        for task in tasks:
            self.tree.insert(task.weight, task)
        self.weigh()

    def weigh(self):
        tasks = self.tree.values()
        for task in tasks:
            task.weight = self.task_weight(task, tasks)
            # time = task.time * task.weight
            self.tree.insert(task.weight, task)

    def by_label(self, label):
        tasks = [(key, task) for key, task in self.tree.items() if label in task.label]
        if tasks:
            return tasks[0]
        return None

    def save(self, path):
        towrite = {"stats": self.stats,
                    "tasks": [dict(vars(task)) for task in self.tree.values()]
                   }
        with open(path, 'w') as fle:
            fle.write(json.dumps(towrite))

    def remove_task(self, task):
        self.tree.remove(task[0])
        self.weigh()

    def start_task(self, task):
        task.start_time = time()

    def pause_task(self, task):
        if not task.paused:
            task.time = (time() - task.start_time) / 60
            del(task.start_time)
            self.weigh()
        else:
            print('Task:', task, 'is not paused at the moment')

    def __repr__(self):
        total = 0
        print(self.stats['done'])
        for weight, task in self.tree.items():
            total += weight
            print(weight, task)

        return str(total)

    def __getitem__(self, name):
        return self.tree[name]


class Task:
    def __init__(self, label='', text='', nice=0, time=45, due_date=None,
                 weight=0, number=0, paused=False, start_time=0):
        self.label = label
        self.text = text
        self.nice = int(nice)
        self.time = int(time)
        self.paused = paused
        self.start_time = start_time
        self.number = number
        if weight:
            self.weight = weight
        else:
            self.weight = 1024 / 1.25**self.nice
        # if not due_date:
        #     self.due_date = datetime.now()
        # else:
        #     self.due_date = datetime.strptime(due_date, '%Y%m%d %H:%M')

    def __repr__(self):
        return '%s - %d - %d' % (self.label, self.nice, self.time)

    def __getitem__(self, name):
        return self.__dict__[name]


def get_tasks():
    with open(path, 'r') as fle:
        tasks = json.load(fle)
    return tasks
