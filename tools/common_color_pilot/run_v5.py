"""v5 실행기: 화면이 3840x2160일 때만 wide_pilot을 시작한다.

wide_pilot.py 자체는 바꾸지 않는다. 바꾸면 이미 시작한 회차의 code_sha256이 달라져
같은 가명 코드로 이어 할 수 없기 때문이다.
"""
import wide_pilot

REQUIRED_SCREEN = (3840, 2160)


class GuardedApp(wide_pilot.App):
    def start(self):
        # 복제 모드 2560x1440이면 AOC 4K 모니터가 화면을 확대해 선이 번진다.
        screen = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        if screen != REQUIRED_SCREEN:
            return self.messagebox.showerror(
                '화면 해상도',
                f'프로그램이 인식한 화면이 {screen[0]}x{screen[1]}입니다.\n'
                f'AOC 모니터를 {REQUIRED_SCREEN[0]}x{REQUIRED_SCREEN[1]}로 맞춘 뒤, '
                '이 창을 닫고 프로그램을 다시 켜세요.\n'
                'Win+P에서 "두 번째 화면만"을 쓰면 안전합니다.')
        return super().start()


if __name__ == '__main__':
    GuardedApp().run()
