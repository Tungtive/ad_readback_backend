if __name__ == "__main__":
    from app import create_app

    app = create_app()
    print(app.config['MONGODB_DB'])
    app.run(debug=True, host="0.0.0.0", port=3000)
