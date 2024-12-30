https://www.uvicorn.org/settings/


if __name__ == '__main__':
  uvicorn.run("autoops.api.app:app",
              host='0.0.0.0',
              port=int(args.port),
              workers=8,
              loop='uvloop')