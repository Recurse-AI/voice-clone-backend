#!/bin/bash

echo "🌐 Cloudflare Setup for ClearVocals"

if [ -z "$1" ]; then
    echo ""
    echo "Usage: ./setup_cloudflare.sh <domain>"
    echo "Example: ./setup_cloudflare.sh clearvocals.ai"
    echo ""
    echo "This will configure:"
    echo "  • Backend: api.clearvocals.ai"
    echo "  • Files:   files.clearvocals.ai (R2)"
    echo "  • Media:   media.clearvocals.ai (R2)"
    exit 0
fi

DOMAIN=$1

cat > .env.proxy << EOF
FRONTEND_URL=https://${DOMAIN}
BACKEND_URL=https://api.${DOMAIN}
PUBLIC_HOST=${DOMAIN}
R2_PUBLIC_URL=https://files.${DOMAIN}
NODE_ENV=production
DEBUG=false
EOF

echo "✅ Configuration saved"
echo ""
echo "🌐 Frontend:  https://${DOMAIN}"
echo "🔧 Backend:   https://api.${DOMAIN}"
echo "📦 Files R2:  https://files.${DOMAIN}"
echo ""
echo "📋 Cloudflare DNS (auto-proxied):"
echo "   CNAME  api    <runpod-domain>"
echo ""
echo "📋 Cloudflare R2 Custom Domains:"
echo "   R2 Dashboard → Bucket → Settings → Add: files.${DOMAIN}"
echo ""
echo "▶ Run: source .env.proxy && ./runpod_setup.sh"

