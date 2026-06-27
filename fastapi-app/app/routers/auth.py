# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app import models, schemas, utils, database, oauth2

router = APIRouter(
    tags=["Authentication"]
)

templates = Jinja2Templates(directory="app/templates")

@router.post("/auth/login", response_model=schemas.Token)
async def api_login(response: Response, user_credentials: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(database.get_db)):
    query = select(models.User).where(models.User.email == user_credentials.username)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Credentials")
        
    if not utils.verify_password(user_credentials.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Credentials")
        
    access_token = oauth2.create_access_token(data={"user_id": str(user.id)})
    
    # Set cookie for browser support
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login")
async def process_login(
    request: Request, 
    username: str = Form(...), 
    password: str = Form(...), 
    db: AsyncSession = Depends(database.get_db)
):
    # Find user by username
    query = select(models.User).where(models.User.username == username)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    # If no user, create one (Vibecode quick-start logic)
    if not user:
        new_user = models.User(
            username=username, 
            email=f"{username}@example.com",
            password_hash=utils.hash_password(password) # Use hashed password
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        user = new_user

    # Verify password
    if utils.verify_password(password, user.password_hash) or user.password_hash == password:
        # Check if onboarding is needed
        target_url = "/news"
        if user.cluster_id is None:
            target_url = "/onboarding"
            
        response = RedirectResponse(url=target_url, status_code=302)
        response.set_cookie(key="session_user", value=username)
        
        # Also create JWT for session
        access_token = oauth2.create_access_token(data={"user_id": str(user.id)})
        response.set_cookie(key="access_token", value=access_token, httponly=True)
        
        return response
    else:
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error_msg": "Sai tên đăng nhập hoặc mật khẩu!"
        })

@router.post("/register")
async def process_register(
    request: Request, 
    username: str = Form(...), 
    email: str = Form(...), 
    password: str = Form(...), 
    db: AsyncSession = Depends(database.get_db)
):
    # Check if user already exists
    query = select(models.User).where((models.User.username == username) | (models.User.email == email))
    result = await db.execute(query)
    if result.scalar_one_or_none():
        return templates.TemplateResponse("register.html", {
            "request": request, 
            "error_msg": "Tên đăng nhập hoặc Email đã tồn tại!"
        })

    try:
        new_user = models.User(
            username=username, 
            email=email, 
            password_hash=utils.hash_password(password)
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        # Log in immediately after registration and go to onboarding
        response = RedirectResponse(url="/onboarding", status_code=302)
        response.set_cookie(key="session_user", value=username)
        
        access_token = oauth2.create_access_token(data={"user_id": str(new_user.id)})
        response.set_cookie(key="access_token", value=access_token, httponly=True)
        
        return response
    except Exception as e:
        print(f"Lỗi đăng ký: {e}")
        return templates.TemplateResponse("register.html", {
            "request": request, 
            "error_msg": "Có lỗi xảy ra khi tạo tài khoản!"
        })
