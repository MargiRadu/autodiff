from itertools import count
from singleton_decorator import singleton


def container():
    return AutoDiffContainer()


@singleton
class AutoDiffContainer:

    def __init__(self):
        self._active_sections = {}
        self._active_section_ids = {}
        self._sections_stack = []

        self._counter = count(0, 1)

    def register(self, active_section):
        as_id = next(self._counter)
        self._active_sections[as_id] = active_section
        self._active_section_ids[active_section] = as_id

    def unregister(self, active_section):
        as_id = self._active_section_ids[active_section]
        del self._active_section_ids[active_section]
        del self._active_sections[as_id]

    def current_section(self):
        tos_id = self._sections_stack[-1]

        while tos_id not in self._active_section_ids:
            self._sections_stack.pop()
            tos_id = self._sections_stack[-1]

        return tos_id
