import json
import tornado.web
import tornado.auth
import urllib.parse


class AuthLoginHandler(tornado.web.RequestHandler, tornado.auth.GoogleOAuth2Mixin):
    async def get(self):
        redirect_uri = urllib.parse.urljoin(
            self.request.protocol + "://" + self.request.host,
            self.reverse_url("google_oauth"),
        )

        if self.get_argument("code", False):
            try:
                access = await self.get_authenticated_user(
                    redirect_uri=redirect_uri, code=self.get_argument("code")
                )
                user = await self.oauth2_request(
                    "https://www.googleapis.com/oauth2/v1/userinfo",
                    access_token=access["access_token"],
                )

                if user.get("email") == "peder.bkoppang@hotmail.no":
                    self.set_signed_cookie("user", json.dumps(user))
                    self.redirect(self.get_argument("next", "/"))
                else:
                    self.set_status(403)
                    self.write("Access denied. You are not an authorized user.")
                    self.finish()

            except tornado.auth.AuthError as e:
                self.set_status(403)
                self.write(f"Google authentication failed: {e}")
                self.finish()

        else:
            await self.authorize_redirect(
                redirect_uri=redirect_uri,
                client_id=self.settings["google_oauth"]["key"],
                scope=["profile", "email"],
                response_type="code",
            )


class AuthLogoutHandler(tornado.web.RequestHandler):
    def get(self):
        self.clear_cookie("user")
        self.redirect("/")