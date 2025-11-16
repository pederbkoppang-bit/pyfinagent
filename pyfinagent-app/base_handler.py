import json
import tornado.web


class BaseHandler(tornado.web.RequestHandler):
    def get_current_user(self):
        user_json = self.get_signed_cookie("user")
        if user_json:
            return json.loads(user_json)
        return None