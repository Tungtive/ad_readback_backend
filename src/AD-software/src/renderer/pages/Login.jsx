import React, { useEffect, useState } from "react";
import { Form, Input, Button, Checkbox, message } from "antd";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import sty from "./login.module.css";

export default function Login() {
  const [form] = Form.useForm();
  const [isLogin, setIsLogin] = useState(true);

  const onFinish = (values) => {};

  const onFinishFailed = (errorInfo) => {
    console.log("Failed:", errorInfo);
  };

  let navigate = useNavigate();

  return (
    <div className={sty.box}>
      <div className={sty.loginBox}>
        <h1 className={sty.h1}>{isLogin ? "Login" : "Registration"}</h1>
        <Form
          form={form}
          name="basic"
          labelCol={{ span: 5 }}
          wrapperCol={{ span: 16 }}
          initialValues={{ remember: true }}
          onFinish={onFinish}
          onFinishFailed={onFinishFailed}
          autoComplete="off"
        >
          <Form.Item
            label="Username"
            name="username"
            rules={[
              { required: true, message: " Please enter your username!" },
            ]}
          >
            <Input id="usernameInput" />
          </Form.Item>

          <Form.Item
            name="pwd"
            label="Password"
            rules={[
              {
                required: true,
                message: "Please enter your password!",
              },
            ]}
            hasFeedback
          >
            <Input.Password id="pwdInput" />
          </Form.Item>

          {!isLogin && (
            <Form.Item
              name="confirmPwd"
              label="Confirm"
              dependencies={["pwd"]}
              hasFeedback
              rules={[
                {
                  required: true,
                  message: "Please confirm your password!",
                },
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (!value || getFieldValue("pwd") === value) {
                      return Promise.resolve();
                    }
                    return Promise.reject(
                      new Error("The two passwords entered are inconsistent!")
                    );
                  },
                }),
              ]}
            >
              <Input.Password id="confirmPwd" />
            </Form.Item>
          )}

          {!isLogin && (
            <Form.Item
              label="Phone number"
              name="phone"
              rules={[
                { required: true, message: "Please enter your phone number!" },
                {
                  validator(_, value) {
                    const pattern = new RegExp("^1[34578][0-9]{9}$", "i");
                    if (pattern.test(value)) {
                      return Promise.resolve();
                    }
                    return Promise.reject(
                      new Error("Please enter a legal phoen number!")
                    );
                  },
                },
              ]}
            >
              <Input id="phone" />
            </Form.Item>
          )}

          {!isLogin && (
            <Form.Item
              label="E-mail"
              name="mail"
              rules={[
                { required: true, message: "Please enter your E-mail!" },
                { type: "email", message: "Please enter a legal E-mail!" },
              ]}
            >
              <Input id="mail" />
            </Form.Item>
          )}

          {isLogin && (
            <Form.Item
              name="remember"
              valuePropName="checked"
              wrapperCol={{ offset: 6, span: 16 }}
            >
              <Checkbox>Remember me</Checkbox>
            </Form.Item>
          )}

          <div
            style={{
              textAlign: "center",
            }}
          >
            {" "}
            <Button
              style={{
                width: "50%",
              }}
              type="primary"
              htmlType="submit"
              onClick={() => {
                const url = "http://192.168.1.106:3002/";
                if (isLogin) {
                  const userName =
                    document.getElementById("usernameInput").value;
                  const pwd = document.getElementById("pwdInput").value;
                  const params = { username: userName, password: pwd };
                  axios
                    .post(url.concat("auth/login"), params)
                    .then((response) => {
                      console.log(response);
                      navigate("/home");
                      message.success("login success");
                      return response;
                    })
                    .catch((error) => {
                      message.warning("username or password error");
                    });
                } else {
                  const userName =
                    document.getElementById("usernameInput").value;
                  const pwd = document.getElementById("pwdInput").value;
                  const confirmPwd =
                    document.getElementById("confirmPwd").value;
                  const phone = document.getElementById("phone").value;
                  const mail = document.getElementById("mail").value;

                  const params = {
                    username: userName,
                    password: pwd,
                    phone_number: phone,
                    email: mail,
                  };
                  axios
                    .post(url.concat("users/register"), params)
                    .then((response) => {
                      console.log(response);
                      setIsLogin(!isLogin);
                      message.success("register success");
                      return response;
                    })
                    .catch((error) => {
                      return error;
                    });
                }
              }}
            >
              {isLogin ? "Login" : "Registration"}
            </Button>
          </div>
        </Form>
        <div
          style={{
            textAlign: "right",
            marginTop: 30,
          }}
        >
          <Button
            onClick={() => {
              if (isLogin === false) {
                form.resetFields();
              }
              setIsLogin(!isLogin);
            }}
            type="link"
          >
            {isLogin
              ? "No account yet? Register！"
              : "Already registered? Login!"}
          </Button>
        </div>
      </div>
    </div>
  );
}
